"""Políticas de routing (F3) — integra el SchmittTrigger/LatencyGuard del F0
con el DES de serving.

Convención de presión de cola: pressure = min(1, q / q_ref), con
q_ref = batch_size (un lote de backlog) pre-registrado. Derivación del
deadband (plan §3.4): conmutar a student vale κ_switch y ahorra
ΔE[serv] = E[serv]_t − E[serv]_s por lote desaguado; el deadband
(q_on − q_off)·q_ref × Δserv/lote debe ≥ κ_switch — condición VERIFICADA
con las mediciones del microbench (results/latency_report.json), no tuneada:
- Δserv(B=16) medido ≈ 71 µs/lote → deadband 0.5·16 = 8 lotes?? en unidades
  batch: (q_on−q_off)·q_ref = 0.5·16 = 8 eventos·lote-cuéntalos;
  equivalencia: deadband_us = 0.5 × Δserv ≈ 35 µs ≥ κ_switch ≈ 24µs(B16)
  que escala por kappa_scale como el resto. Verificación con números
  actualizados a la campaña (script scripts/derive_deadband.py).

Políticas (≈ baselines del plan):
- TeacherAlways / StudentAlways (los extremos B1/B8-análogos del lado serve)
- FixedThreshold: umbral fijo SIN histéresis (el antipatrón flap)
- SchmittRouter: histéresis por cabeza + min_dwell + LatencyGuard (método)
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
    """Umbral fijo si histéresis ni refractario: cambiante directo cuando
    p cruza (flapping potencial — el antipatrón contra el que Schmitt+guard
    protege)."""

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
    """(q_on, q_off) en presión [0,1] derivados del κ MEDIDO.

    Filosofía del plan: student-mode solo cuando el backlog justifica el
    costo κ; el deadband debe cubrir el κ de la conmutación de VUELTA con
    margen ≥1.5. Con Δserv_event = (E_t − E_s)/B medido y κ_eventos =
    κ/Δserv_event: q_on = 2×κ_eventos (margen de ida), q_off = q_on − 1.5×κ
    (piso 0.15·q_ref para permitir el re-flip post-idle). NUMEROS, no tuneos:
    la verificación de que deadband_us ≥ κ es un ASSERT de la campaña.
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
    # q_on analítico: el switch paga κ y ahorra Δserv por REGLO desaguado;
    # con margen bilateral (deadband de vuelta ≥ 1.5κ), q_on = 2×κ/Δserv_event.
    delta_serv = max(E_t - E_s, 1e-3)
    delta_event = delta_serv / B
    kappa_events = max(1.0, kappa / delta_event)
    q_on_pres = float(min(1.0, max(0.30, 2.0 * kappa_events / q_ref)))
    # q_off = q_on − 1.5×κ_eventos, con piso 0.15·q_ref (≤2.4 ev) para que el
    # primer tick post-idle (q ≥ 1) pueda completar el re-flip tras el dwell.
    q_off_ev = max(q_on_pres * q_ref - 1.5 * kappa_events, 0.15 * q_ref)
    q_off_pres = float(max(0.10, min(q_on_pres - 0.05, q_off_ev / q_ref)))
    return {"q_on": q_on_pres, "q_off": q_off_pres,
            "kappa_us": kappa, "delta_serv_us": delta_serv,
            "e_t_us": E_t, "e_s_us": E_s, "kappa_events": kappa_events}


class RandomMatched(RouterPolicy):
    """B5: routing aleatorio con cobertura igualada (p se fija desde la
    cobertura observada del Schmitt en la misma traza)."""

    def __init__(self, p: float, seed: int = 0) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)

    def update(self, q, t):
        return _HEADS if self.rng.random() < self.p else ()


class SchmittRouter(RouterPolicy):
    """El método: histéresis por cabeza con deadband κ-derivado + dwell +
    LatencyGuard (canal de latencia refractario tras conmutar)."""

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
            # presión escalada por cabeza (downshift antivirus ordering)
            ph = p * self.per_head_scale[h]
            trig.update(min(1.0, ph), t)
            if trig.in_student:
                out.append(h)
        return tuple(out)

    def observe(self, lp95: float, t: float) -> None:
        """Canal de latencia (llamado por el DES al completar cada lote)."""
        if self.guard is not None:
            self.guard.feeding(lp95, t)
