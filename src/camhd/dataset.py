"""Construction of replay datasets from the coupled generator.

Splits by time horizon with seeds separated by subsystem
(RunConfig v3): each block is a generator's own horizon.
Feature standardization is set ONLY in train (train-only stats).

The state of charge s of each event: 0 = normal, 1 = burst (proxy of the
stratum; DES will add actual q_t quantiles for stratum per queue).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import StreamConfig
from .generator import CoupledStream, StreamSample

__all__ = ["ReplayDataset", "build_replay"]


@dataclass
class ReplayDataset:
    features: torch.Tensor        # [N, d] float32 estandarizado
    gate: torch.Tensor            # [N] int64
    choice: torch.Tensor          # [N] int64
    severity: torch.Tensor        # [N] int64 (0 benigno; 1..C−1 ataque)
    stratum: torch.Tensor         # [N] int64 (0 normal, 1 burst)
    t: torch.Tensor               # [N] float64 (llegadas)
    standardize: tuple[np.ndarray, np.ndarray] | None = None  # (mu, sd) train-only
    s_vec: torch.Tensor | None = None  # [N,3] S_infra (q,λ,ℓ_p95) — del DES glass box

    def __len__(self) -> int:
        return self.features.size(0)

    @property
    def s(self) -> torch.Tensor:
        if self.s_vec is None:
            return torch.zeros(len(self), 3, device=self.features.device)
        return self.s_vec

    def to(self, device) -> "ReplayDataset":
        return ReplayDataset(
            features=self.features.to(device),
            gate=self.gate.to(device),
            choice=self.choice.to(device),
            severity=self.severity.to(device),
            stratum=self.stratum.to(device),
            t=self.t.to(device),
            standardize=self.standardize,
            s_vec=None if self.s_vec is None else self.s_vec.to(device),
        )


def _standardize(x: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None):
    """Z-score con stats train-only; retorna (x_norm, (mu, sd))."""
    if stats is None:
        mu = x.mean(0, keepdims=True)
        sd = np.maximum(x.std(0, keepdims=True), 1e-6)
    else:
        mu, sd = stats
    return (x - mu) / sd, (mu, sd)


def _samples_to_tensors(sample: StreamSample, stats) -> tuple:
    x = sample.features.astype(np.float32)
    x_norm, new_stats = _standardize(x, stats)
    return (
        torch.from_numpy(x_norm.astype(np.float32)),
        torch.from_numpy(sample.gate.astype(np.int64)),
        torch.from_numpy(sample.choice.astype(np.int64)),
        torch.from_numpy(sample.severity.astype(np.int64)),
        torch.from_numpy(sample.in_burst.astype(np.int64)),
        torch.from_numpy(sample.t.astype(np.float64)),
        new_stats,
    )


def build_replay(horizons: dict[str, float], stream_cfg: StreamConfig,
                 seeds: dict[str, int] | None = None) -> dict[str, ReplayDataset]:
    """Genera los horizontes (train/val/test) con seed por split."""
    seed_map = seeds or {"train": 0, "val": 1, "test": 2}
    cfg_fields = stream_cfg.__dict__
    out: dict[str, ReplayDataset] = {}
    stats: tuple[np.ndarray, np.ndarray] | None = None
    for name, horizon in horizons.items():
        cfg = StreamConfig(**{**cfg_fields, "seed": seed_map[name]})
        sample = CoupledStream(cfg).generate_until(horizon)
        feats, gate, choice, severity, stratum, t, stats = _samples_to_tensors(sample, stats)
        out[name] = ReplayDataset(features=feats, gate=gate, choice=choice,
                                  severity=severity, stratum=stratum, t=t,
                                  standardize=stats)
    return out
