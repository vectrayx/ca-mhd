"""DES de serving — Phase F2 (plan v3 §4: CUL-Bench engine).

Discrete event simulation of an MLP serving core, ON THIS CPU:

- Arrivals: exact from the coupled generator (same law as training).
- Service by MICRO-LOTS: when the core is free, it drains the tail until
B events and executes A real forward (`forward_heads`) — the time of
service is the MEASURED latency of that batch on this CPU × `service_scale`
(anchor to pre-registered operating point: ρ_t(burst)≈1.1 > 1 ≥ ρ_s).
No synthetic distributions: the jitter is the actual hardware jitter.
- κ_switch: when the set of narrow heads switches between services
consecutive, the measured excess is billed (eviction of 16 MB + cold fetch).
- Router: external policy; receives q (backlog to the tick) and returns the
set of heads in student mode. Schmitt/hysteresis live there.
- S_infra = (q_t, λ_t, ℓ_p95_t) recorded by event (model input,
contract stratification δ(s)).

Training replay (glass box) = same DES with fixed policy
all-teacher: real saturation in bursts → deep queue states.

Timing: ALL in "stream seconds"; latencies reported in µs ×1e6.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models import CAMHDNet
from ..generator import StreamSample
from .latency import LatencyModel

__all__ = ["SimConfig", "SimResult", "RouterPolicy", "run_des", "glass_box_des"]


@dataclass
class SimConfig:
    batch_size: int = 16
    service_scale: float = 2.5      # ancla del punto operativo (ver docstring)
    kappa_scale: float = 2.5        # same core → same factor
    rate_idle: float = 2000.0       # arrivals/s (recorded from StreamConfig)
    rate_burst: float = 14000.0
    t_end: float | None = None      # None = exhaust the stream
    l_max_us: float = 1500.0        # SLA p99 per event
    device: str = "cpu"
    lp95_window: int = 200          # ℓ_p95 rolling window (for S_infra)
    max_services: int = 2_000_000   # bodyguard
    # S_infra scaling FOR THE MODEL (same pipeline as training;
    # the stored s_vec are left RAW for stratification)
    s_scale: tuple[float, float, float] = (200.0, 14000.0, 1e4)
    # service clock seed (Amendment 3 + repro by deploy): run_des
    # RESET the RNG of the LatencyModel with this seed — the dispatch order
    # of the campaign should not affect the metrics of a deploy
    rng_seed: int = 7420


@dataclass
class SimResult:
    t_arr: np.ndarray          # arrival [n] (s of stream)
    t_dep: np.ndarray          # game [n] (stream s)
    q_arr: np.ndarray          # queue observed upon arrival (pre-join, head-of-line)
    q_s: np.ndarray            # backlog to service tick (router input)
    lat_us: np.ndarray         # (t_dep − t_arr) ×1e6
    s_vec: np.ndarray          # [n,3] (q_t, λ_t, ℓ_p95_t) to the service
    narrow: np.ndarray         # [n,3] bool per head (gate, choice, corn)
    kappa_n: np.ndarray        # [n] 1 if the service consumed a κ_switch
    pred_gate: np.ndarray      # [n] attack prob deployed
    pred_choice: np.ndarray    # [n] argmax choice
    pred_level: np.ndarray     # [n] argmax pmf corn
    conf_choice: np.ndarray    # [n] confianza top-1 choice
    conf_corn: np.ndarray      # [n] confianza top-1 corn (pmf)
    n_switches: int = 0
    summary: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t_arr)

    def summarize(self, l_max_us: float) -> dict:
        lat = self.lat_us
        return {
            "n_events": int(self.n),
            "n_switches": int(self.n_switches),
            "lat_mean_us": float(lat.mean()),
            "lat_p95_us": float(np.percentile(lat, 95)) if self.n else float("nan"),
            "lat_p99_us": float(np.percentile(lat, 99)) if self.n else float("nan"),
            "sla_viol_rate": float((lat > l_max_us).mean()) if self.n else float("nan"),
            "q_mean": float(self.q_arr.mean()),
            "q_p95": float(np.percentile(self.q_arr, 95)) if self.n else float("nan"),
            "q_max": int(self.q_arr.max()) if self.n else 0,
        }


class RouterPolicy:
    """Base policy: all heads in teacher mode ALWAYS.

update(q, t) → subset of ("gate","choice","corn") in student mode.
Stateful (Schmitt) policies keep your internal machine here.
    """

    def update(self, q: int, t: float) -> tuple[str, ...]:
        return ()


def run_des(
    sample: StreamSample,
    model: CAMHDNet | None,
    policy: RouterPolicy | None,
    lat: LatencyModel,
    cfg: SimConfig,
    standardize: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimResult:
    """DES of serving. policy=None → all-teacher (glass box); model=None →
pure queue service (only harness sanity without inference)."""
    # Service clock reproducible BY DEPLOY: reset the RNG of the
    # LatencyModel (lognormal with measured CV, fixed seed by deploy —
    # The dispatch order of the campaign is irrelevant for the
    # metrics; Amendment 3).
    lat.rng = np.random.default_rng(cfg.rng_seed)
    HN = ("gate", "choice", "corn")
    n_src = len(sample)
    t_arr = sample.t
    feats = sample.features
    if standardize is not None:
        mu, sd = standardize
        feats = (feats - mu) / np.maximum(sd, 1e-6)
    if cfg.t_end is not None:
        keep = t_arr <= cfg.t_end
        n = int(keep.sum())
    else:
        n = n_src
    t_arr = np.asarray(t_arr[:n], dtype=np.float64)

    evict = torch.empty(4 << 20, dtype=torch.float32)
    device = torch.device(cfg.device)
    model = model.eval() if model is not None else None

    # event departures
    t_dep = np.zeros(n, dtype=np.float64)
    q_arr = np.zeros(n, dtype=np.int64)
    q_s = np.zeros(n, dtype=np.int64)
    lat_us = np.zeros(n, dtype=np.float64)
    s_vec = np.zeros((n, 3), dtype=np.float32)
    narrow_mat = np.zeros((n, 3), dtype=bool)
    kappa_n = np.zeros(n, dtype=np.int8)
    pred_gate = np.zeros(n, dtype=np.float32)
    pred_choice = np.zeros(n, dtype=np.int64)
    pred_level = np.zeros(n, dtype=np.int64)
    conf_choice = np.zeros(n, dtype=np.float32)
    conf_corn = np.zeros(n, dtype=np.float32)
    n_switch = 0

    lp95_win: deque[float] = deque(maxlen=cfg.lp95_window)
    queue: deque[int] = deque()

    # server status
    serv_end: float | None = None
    serv_batch: tuple[int, ...] = ()
    serv_narrow: tuple[str, ...] = ()
    serv_dur = 0.0
    cur_narrow: tuple[str, ...] | None = None
    i = 0
    done = 0
    steps = 0

    def _start_service(t_start: float) -> None:
        nonlocal serv_end, serv_batch, serv_narrow, serv_dur, cur_narrow, n_switch
        q_now = len(queue)
        if policy is not None:
            narrow = tuple(policy.update(q_now, t_start))
        else:
            narrow = ()
        kappa_here = (cur_narrow is not None and narrow != cur_narrow)
        if kappa_here:
            evict.fill_(float(done))   # cold fetch of the new set of weights
            n_switch += 1
        b = min(cfg.batch_size, len(queue))
        idx = tuple(queue.popleft() for _ in range(b))
        lam = float(cfg.rate_burst if sample.in_burst[idx[0]] else cfg.rate_idle)
        lp95 = float(np.percentile(np.asarray(lp95_win, dtype=np.float64), 95)) if lp95_win else 0.0
        # batch event log
        for j in idx:
            s_vec[j] = (q_now, lam, lp95)
            for k, name in enumerate(HN):
                narrow_mat[j, k] = name in narrow
            kappa_n[j] = 1 if kappa_here else 0
        # Amendment 3 (2026-10-21): the service clock leaves the table of the
        # microbench (lognormal sampling with measured CV, fixed seed) and κ
        # switching is inserted as a separate cost. The real forward runs
        # to get the PREDICTIONS, off the clock (same as the
        # post-process): live machine time is not reproducible and
        # drags staging overheads that the microbench amortizes — which
        # moved the operating point ρ between runs (pilot v1 bug:
        # the queue was not draining because the live service was measuring ~50 ms/event).
        dur_us = 0.0
        if len(narrow) == 0:
            dur_us = float(lat.sample_service_us("teacher", b))
        elif len(narrow) == 3:
            dur_us = float(lat.sample_service_us("student", b))
        else:
            dur_us = 0.5 * (float(lat.sample_service_us("teacher", b))
                            + float(lat.sample_service_us("student", b)))
        if kappa_here:
            dur_us += float(lat.sample_kappa_us(b))
        if model is not None:
            # PREDICTIONS from the actual served model (off-clock)
            xb = torch.from_numpy(np.ascontiguousarray(feats[list(idx)]).astype(np.float32))
            s_row = np.asarray([[q_now / cfg.s_scale[0], lam / cfg.s_scale[1],
                                 lp95 / cfg.s_scale[2]]], dtype=np.float32)
            sb = torch.from_numpy(np.tile(s_row, (b, 1)))
            with torch.no_grad():
                if len(narrow) == 0:
                    out = model(xb, sb, mode="teacher")["t"]
                elif len(narrow) == 3:
                    out = model(xb, sb, mode="student")["s"]
                else:
                    out = model.forward_heads(xb, sb, narrow_heads=narrow)
                pred_gate[list(idx)] = torch.softmax(out["gate"], -1)[:, 1].numpy()
                pred_choice[list(idx)] = out["choice"].argmax(-1).numpy()
                probs_c = torch.softmax(out["choice"], -1)
                conf_choice[list(idx)] = probs_c.max(dim=-1).values.numpy()
            from ..losses import corn_pmf
            with torch.no_grad():
                pmf = corn_pmf(out["corn"])
            pred_level[list(idx)] = pmf.argmax(-1).numpy()
            conf_corn[list(idx)] = pmf.max(dim=-1).values.numpy()
        dur_stream = (dur_us * cfg.service_scale) / 1e6
        serv_end = t_start + dur_stream
        serv_batch = idx
        serv_narrow = narrow
        serv_dur = dur_stream
        cur_narrow = narrow

    while done < n and steps < cfg.max_services:
        steps += 1
        if serv_end is not None and (i >= n or serv_end <= t_arr[i]):
            # complete service
            t_clock = serv_end
            for j in serv_batch:
                t_dep[j] = t_clock
                lat_us[j] = (t_clock - t_arr[j]) * 1e6
                lp95_win.append(float(lat_us[j]))
            done += len(serv_batch)
            # channel latency policy (LatencyGuard method)
            if queue and policy is not None and hasattr(policy, "observe"):
                lp95_now = float(np.percentile(np.asarray(lp95_win, dtype=np.float64), 95))
                policy.observe(lp95_now, t_clock)
            # siguiente servicio inmediato si hay cola
            serv_end = None
            if queue:
                _start_service(t_clock)
        elif i < n:
            # arrival (or batch of simultaneous arrivals)
            t_clock = t_arr[i]
            while i < n and t_arr[i] == t_clock:
                q_arr[i] = len(queue)
                queue.append(i)
                i += 1
            if serv_end is None:
                _start_service(t_clock)
        else:
            break

    if done < n:
        raise RuntimeError(f"DES incompleto: done={done}/{n} (steps={steps})")

    res = SimResult(
        t_arr=t_arr, t_dep=t_dep, q_arr=q_arr, q_s=q_s, lat_us=lat_us,
        s_vec=s_vec, narrow=narrow_mat, kappa_n=kappa_n,
        pred_gate=pred_gate, pred_choice=pred_choice, pred_level=pred_level,
        conf_choice=conf_choice, conf_corn=conf_corn, n_switches=n_switch,
    )
    res.summary = res.summarize(cfg.l_max_us)
    return res


def glass_box_des(
    sample: StreamSample,
    model: CAMHDNet | None,
    cfg: SimConfig,
    lat: LatencyModel,
    standardize: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimResult:
    """Training replay: fixed all-teacher policy."""
    return run_des(sample, None, None, lat, cfg, standardize) if model is None else \
        run_des(sample, model, None, lat, cfg, standardize)
