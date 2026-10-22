"""Latency and κ_switch microbench on the target hardware (plan v3 G4/R5).

Service model, 100% measured on THIS CPU (1 thread, eager torch — the
serving runtime): the deployment server works by MICRO-BATCHES
(typical in serving batch-1 MLP). Measurement table:

- by batch B ∈ {1,4,8,16,32} × mode: average/p95/p99/CV of the actual forward
(µs per BATCH, with per-event derived). DES interpolates this table to
the service time of the batches.
- `per_event` shows the teacher/student gap popup (only computation when
batching amortizes dispatch: factor ~1.3× at B=32, measured).
- `kappa_switch_us[B]`: excess of the alternating pair teacher→student (with
eviction of 16 MB between modes — cold cache/weight TLB) vs steady.
- `jitter_cv`: CV of the execution noise for the DES distribution.

The anchor SLA: L_max = 1.5 ms p99 per event, ≥5k ev/s per core.
Honesty: eager end-to-end batch=1 are dispatch dominance (~350µs);
The paper capacity phenomenon lives in the batched regime, which is the
DES serving regime.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..models import CAMHDNet, ArchConfig

__all__ = ["run_latency_bench", "LatencyModel", "BATCH_GRID"]

BATCH_GRID = (1, 4, 8, 16, 32)


def _time(fn, warm: int, meas: int) -> np.ndarray:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        out = np.empty(meas)
        for i in range(meas):
            t0 = time.perf_counter_ns()
            fn()
            out[i] = (time.perf_counter_ns() - t0) / 1000.0  # µs
    return out


def _stats(times: np.ndarray) -> dict[str, float]:
    return {"mean": float(times.mean()),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "cv": float(times.std() / max(times.mean(), 1e-9))}


def _interp(table: dict[int, dict[str, float]], b: int, key: str) -> float:
    bs = sorted(table)
    if b <= bs[0]:
        return table[bs[0]][key]
    if b >= bs[-1]:
        return table[bs[-1]][key]
    import bisect
    i = bisect.bisect_left(bs, b)
    b0, b1 = bs[i - 1], bs[i]
    w = (b - b0) / (b1 - b0)
    return float(table[b0][key] * (1 - w) + table[b1][key] * w)


def run_latency_bench(net: CAMHDNet | None = None, d: int = 16,
                      n_warmup: int = 200, n_meas: int = 1200, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    if net is None:
        net = CAMHDNet(ArchConfig(w=128, depth=2, d=d)).eval()
    n_warm = max(50, n_warmup // 2)
    n_meas = min(n_meas, 600)
    x = torch.randn(BATCH_GRID[-1], net.cfg.d)
    s = torch.randn(BATCH_GRID[-1], net.cfg.s_dim)

    batch_table: dict[str, dict[int, dict[str, float]]] = {
        "teacher": {}, "student": {}}
    kappa_by_batch: dict[int, float] = {}
    for B in BATCH_GRID:
        xb, sb = x[:B], s[:B]
        t_t = _time(lambda: net(xb, sb, mode="teacher"), n_warm, n_meas)
        t_s = _time(lambda: net(xb, sb, mode="student"), n_warm, n_meas)
        batch_table["teacher"][B] = _stats(t_t)
        batch_table["student"][B] = _stats(t_s)
        # κ_switch: alternation with eviction of 16MB between modes (cold weights).
        # fill_ = pure memory traffic (no RNG cost), forces cold-fetch.
        evict = torch.empty(4 << 20, dtype=torch.float32)  # 16 MB
        with torch.no_grad():
            for i in range(n_warm // 4):
                evict.fill_(float(i))
                net(xb, sb, mode="teacher")
                evict.fill_(float(i) + 0.5)
                net(xb, sb, mode="student")
            ts = []
            mode = "teacher"
            for i in range(n_meas):
                evict.fill_(float(i))
                t0 = time.perf_counter_ns()
                net(xb, sb, mode=mode)
                ts.append((time.perf_counter_ns() - t0) / 1000.0)
                mode = "student" if mode == "teacher" else "teacher"
            t_alt = np.asarray(ts)
        # half of the alternate calls are EACH mode → its pair
        # steady equivalent is (mean_t + mean_s)/2 per call
        steady_half = (batch_table["teacher"][B]["mean"]
                       + batch_table["student"][B]["mean"]) / 2
        alt_mean = float(np.median(t_alt))
        kappa_by_batch[B] = max(0.0, alt_mean - steady_half)

    flops = net.flops_forward(BATCH_GRID[-1])
    return {
        "batch_table": batch_table,
        "per_event_table": {
            mode: {B: {k: v / B for k, v in st.items()}
                   for B, st in tab.items()}
            for mode, tab in batch_table.items()},
        "kappa_switch_us": kappa_by_batch,
        "flops_per_forward_B32": flops,
        "eager_B1_end_to_end": {
            "teacher": batch_table["teacher"][1],
            "student": batch_table["student"][1],
        },
        "sl_anchor": {"L_max_p99_us": 1500.0, "min_evt_per_s": 5000.0},
        "arch": {"w": net.cfg.w, "depth": net.cfg.depth,
                 "student_trunk_frac": net.cfg.student_trunk_frac, "d": net.cfg.d,
                 "head_mult": net.cfg.head_mult},
        "cpu_threads": torch.get_num_threads(),
        "batch_grid": list(BATCH_GRID),
    }


class LatencyModel:
    """DES service model: batch×mode measurement table + κ_switch."""

    def __init__(self, report: dict, batch_size: int = 16, rng_seed: int = 777) -> None:
        self.batch_size = batch_size
        self.table_t = {int(k): v for k, v in report["batch_table"]["teacher"].items()}
        self.table_s = {int(k): v for k, v in report["batch_table"]["student"].items()}
        self.kappa_by_batch = {int(k): float(v) for k, v in report["kappa_switch_us"].items()}
        self.rng = np.random.default_rng(rng_seed)

    def mean_service_us(self, mode: str, b: int | None = None) -> float:
        b = b or self.batch_size
        tab = self.table_t if mode == "teacher" else self.table_s
        return _interp(tab, b, "mean")

    def cv_service(self, mode: str, b: int | None = None) -> float:
        b = b or self.batch_size
        tab = self.table_t if mode == "teacher" else self.table_s
        return _interp(tab, b, "cv")

    def sample_service_us(self, mode: str, b: int | None = None) -> float:
        """BATCH time of b events: lognormal with measured cv, truncated ≥ 1µs."""
        b = b or self.batch_size
        mean = self.mean_service_us(mode, b)
        cv = self.cv_service(mode, b)
        s2_log = float(np.log1p(min(cv, 0.5) ** 2))
        mu = float(np.log(mean) - s2_log / 2)
        return float(max(self.rng.lognormal(mu, np.sqrt(s2_log)), 1.0))

    def sample_kappa_us(self, b: int | None = None) -> float:
        """Mode switching penalty (cold fetch of the other set)."""
        b = b or self.batch_size
        bs = sorted(self.kappa_by_batch)
        if b <= bs[0]:
            k = self.kappa_by_batch[bs[0]]
        elif b >= bs[-1]:
            k = self.kappa_by_batch[bs[-1]]
        else:
            import bisect
            i = bisect.bisect_left(bs, b)
            b0, b1 = bs[i - 1], bs[i]
            w = (b - b0) / (b1 - b0)
            k = self.kappa_by_batch[b0] * (1 - w) + self.kappa_by_batch[b1] * w
        s2_log = float(np.log1p(0.5 ** 2))
        mu = float(np.log(max(k, 1e-3)) - s2_log / 2)
        return float(max(self.rng.lognormal(mu, np.sqrt(s2_log)), 0.0))

    @classmethod
    def from_json(cls, path, batch_size: int = 16, rng_seed: int = 777) -> "LatencyModel":
        with open(path) as f:
            return cls(json.load(f), batch_size=batch_size, rng_seed=rng_seed)
