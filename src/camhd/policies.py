"""Routing policies (F3) — integrates the SchmittTrigger/LatencyGuard of F0
with the DES of serving.

Tail pressure convention: pressure = min(1, q / q_ref), with
q_ref = batch_size (a backlog batch) pre-registered. Derivation of
deadband (plan §3.4): switching to student is worth κ_switch and saves
ΔE[serv] = E[serv]_t − E[serv]_s per dewatered batch; the deadband
(q_on − q_off)·q_ref × Δserv/batch must ≥ κ_switch — VERIFIED condition
with microbench measurements (results/latency_report.json), not tuned:
- Δserv(B=16) measured ≈ 71 µs/batch → deadband 0.5·16 = 8 batches?? in units
batch: (q_on−q_off)·q_ref = 0.5·16 = 8 events·batch-count them;
equivalence: deadband_us = 0.5 × Δserv ≈ 35 µs ≥ κ_switch ≈ 24µs(B16)
which scales by kappa_scale like the rest. Verification with numbers
updated to the campaign (script scripts/derive_deadband.py).

Policies (≈ plan baselines):
- TeacherAlways / StudentAlways (the B1/B8-analog ends of the serve side)
- FixedThreshold: fixed threshold WITHOUT hysteresis (the flap anti-pattern)
- SchmittRouter: hysteresis per head + min_dwell + LatencyGuard (method)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RouterConfig
from .router import LatencyGuard, Mode, SchmittTrigger
from .bench.sim import RouterPolicy

_HEADS = ("gate", "choice", "corn")
__all__ = [
    "TeacherAlways", "StudentAlways", "FixedThreshold", "RandomMatched",
    "SchmittRouter", "derive_thresholds_from_latency",
]


class TeacherAlways(RouterPolicy):
    def update(self, q, t):
        return ()


class StudentAlways(RouterPolicy):
    def update(self, q, t):
        return _HEADS


class FixedThreshold(RouterPolicy):
    """Fixed threshold if hysteresis or refractory: directly changing when
p crosses (potential flapping — the anti-pattern against which Schmitt+guard
Protect """

    def __init__(self, q_ref: float = 16.0, q_thresh: float = 0.5) -> None:
        self.q_ref = q_ref
        self.q_thresh = q_thresh

    def update(self, q, t):
        p = min(1.0, q / self.q_ref)
        return _HEADS if p >= self.q_thresh else ()


def derive_thresholds_from_latency(
    lat_report: dict, q_ref: float = 16.0, kappa_scale: float = 2.5,
    service_scale: float = 2.5,
) -> dict[str, float]:
    """(q_on, q_off) in pressure [0,1] derivatives of the MEASURED κ.

Plan philosophy: student-mode only when the backlog justifies the
cost κ; The deadband must cover the RETURN switching κ with
margin ≥1.5. With Δserv_event = (E_t − E_s)/B measured and κ_events =
κ/Δserv_event: q_on = 2×κ_events (one-way margin), q_off = q_on − 1.5×κ
(floor 0.15·q_ref to allow post-idle re-flip). NUMBERS, not tunes:
verifying that deadband_us ≥ κ is an ASSERT of the campaign.
    """
    import bisect

    def interp(tab: dict[int, dict], b: int, key: str) -> float:
        bs = sorted(tab)
        if b <= bs[0]:
            return tab[bs[0]][key]
        if b >= bs[-1]:
            return tab[bs[-1]][key]
        i = bisect.bisect_left(bs, b)
        b0, b1 = bs[i - 1], bs[i]
        w = (b - b0) / (b1 - b0)
        return tab[b0][key] * (1 - w) + tab[b1][key] * w

    tab_t = {int(k): v for k, v in lat_report["batch_table"]["teacher"].items()}
    tab_s = {int(k): v for k, v in lat_report["batch_table"]["student"].items()}
    B = 16
    E_t = interp(tab_t, B, "mean") * service_scale
    E_s = interp(tab_s, B, "mean") * service_scale
    kappa_by_b = {int(k): float(v) for k, v in lat_report["kappa_switch_us"].items()}
    kappa_key = kappa_by_b[B]
    kappa = max(kappa_key, 1.0) * kappa_scale
    # q_on analytical: the switch pays κ and saves Δserv per dewatered rule;
    # with bilateral margin (return deadband ≥ 1.5κ), q_on = 2×κ/Δserv_event.
    delta_serv = max(E_t - E_s, 1e-3)
    delta_event = delta_serv / B
    kappa_events = max(1.0, kappa / delta_event)
    q_on_pres = float(min(1.0, max(0.30, 2.0 * kappa_events / q_ref)))
    # q_off = q_on − 1.5×κ_eventos, with floor 0.15·q_ref (≤2.4 ev) so that the
    # first post-idle tick (q ≥ 1) can complete the re-flip after the dwell.
    q_off_ev = max(q_on_pres * q_ref - 1.5 * kappa_events, 0.15 * q_ref)
    q_off_pres = float(max(0.10, min(q_on_pres - 0.05, q_off_ev / q_ref)))
    return {"q_on": q_on_pres, "q_off": q_off_pres,
            "kappa_us": kappa, "delta_serv_us": delta_serv,
            "e_t_us": E_t, "e_s_us": E_s, "kappa_events": kappa_events}


class RandomMatched(RouterPolicy):
    """B5: Random routing with equal coverage (p is set from the
observed Schmitt coverage in the same trace)."""

    def __init__(self, p: float, seed: int = 0) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)

    def update(self, q, t):
        return _HEADS if self.rng.random() < self.p else ()


class SchmittRouter(RouterPolicy):
    """The method: head hysteresis with κ-derivative deadband + dwell +
LatencyGuard (refractory latency channel after switching)."""

    def __init__(self, rc: RouterConfig, q_ref: float = 16.0,
                 per_head_scale: dict[str, float] | None = None,
                 use_latency_guard: bool = True) -> None:
        self.q_ref = q_ref
        self.triggers = {
            h: SchmittTrigger(cfg=rc, name=h) for h in _HEADS
        }
        self.guard = LatencyGuard(rc) if use_latency_guard else None
        self.use_latency_guard = use_latency_guard
        self.per_head_scale = per_head_scale or {h: 1.0 for h in _HEADS}
        self._q_ref_cfg = rc

    def update(self, q, t) -> tuple[str, ...]:
        p = min(1.0, q / self.q_ref)
        out = []
        for h in _HEADS:
            trig = self.triggers[h]
            # scaled pressure per head (downshift antivirus ordering)
            ph = p * self.per_head_scale[h]
            trig.update(min(1.0, ph), t)
            if trig.in_student:
                out.append(h)
        return tuple(out)

    def observe(self, lp95: float, t: float) -> None:
        """Latency channel (called by the DES upon completion of each batch)."""
        if self.guard is not None:
            self.guard.feeding(lp95, t)
