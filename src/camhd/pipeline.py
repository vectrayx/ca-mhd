"""Pipeline F2: streams → DES glass box → replay con S_infra → teacher.

- splits por horizonte, seeds separadas por subsistema (RunConfig).
- El DES glass box (all-teacher, servicio por tabla medida del microbench)
  materializa la trayectoria de cola q_t que define los estratos de carga y
  el estado S_infra = (q, λ, ℓ_p95) que condiciona el trunk.
- Escalado de S_infra pre-registrado: q/q_scale, λ/λ_scale, ℓ_p95/ℓ_scale.
- Estratos por cuantiles de q del train glass box: calm (≤q50),
  warm (50–90), hot (>90). El estrato viaja en `stratum` y el G1 se evalúa
  smoothECE por cabeza × estrato.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import RunConfig, StreamConfig
from .dataset import ReplayDataset
from .generator import CoupledStream, StreamSample
from .bench.latency import LatencyModel
from .bench.sim import SimConfig, SimResult, glass_box_des

__all__ = ["SInfraScale", "ReplayBundle", "build_replay_bundle"]


@dataclass(frozen=True)
class SInfraScale:
    q_scale: float = 200.0        # backlog p95 esperado ~1e2–1e3 → /200
    lam_scale: float = 14000.0    # rate_burst del punto operativo
    lp95_scale: float = 1e4       # ℓ_p95 del núcleo edge: 10 ms

    def apply(self, s_vec: np.ndarray) -> np.ndarray:
        return np.stack([
            s_vec[:, 0] / self.q_scale,
            s_vec[:, 1] / self.lam_scale,
            s_vec[:, 2] / self.lp95_scale,
        ], axis=1).astype(np.float32)


@dataclass
class ReplayBundle:
    datasets: dict[str, ReplayDataset]
    glass: dict[str, SimResult]
    samples: dict[str, StreamSample]
    q_tiles: tuple[float, float]   # (q50, q90) del train glass box


def _stratum_from_q(q: np.ndarray, tiles: tuple[float, float]) -> np.ndarray:
    return np.digitize(q, bins=tiles, right=True).astype(np.int64)  # 0,1,2


def build_replay_bundle(
    horizons: dict[str, float],
    stream_cfg: StreamConfig,
    sim_cfg: SimConfig,
    lat: LatencyModel,
    seeds: dict[str, int] | None = None,
    scale: SInfraScale | None = None,
) -> ReplayBundle:
    """Genera cada split con seed propia y ejecuta el DES glass box (all
    teacher) para materializar los estados de carga del replay."""
    seed_map = seeds or {"train": 0, "val": 1, "test": 2, "rollout": 3}
    scale = scale or SInfraScale()
    datasets: dict[str, ReplayDataset] = {}
    glass: dict[str, SimResult] = {}
    samples: dict[str, StreamSample] = {}
    stats: tuple[np.ndarray, np.ndarray] | None = None
    q_tiles = (1.0, 2.0)

    # primer split (train) fija la estandarización y los cuantiles
    names = ["train"] + [k for k in horizons if k != "train"]
    for name in names:
        horizon = horizons[name]
        sc = StreamConfig(**{**stream_cfg.__dict__, "seed": seed_map.get(name, seed_map["train"])})
        stream = CoupledStream(sc).generate_until(horizon)
        samples[name] = stream
        gp = glass_box_des(stream, None, sim_cfg, lat, None)
        glass[name] = gp
        if name == "train":
            x = stream.features.astype(np.float32)
            mu = x.mean(0, keepdims=True)
            sd = np.maximum(x.std(0, keepdims=True), 1e-6)
            stats = (mu, sd)
            q50, q90 = np.percentile(gp.s_vec[:, 0], [50, 90])
            q_tiles = (float(q50), float(q90))
        s_scaled = torch.from_numpy(scale.apply(gp.s_vec))
        stratum = _stratum_from_q(gp.s_vec[:, 0], q_tiles)
        gn = (stream.features - stats[0]) / stats[1]
        datasets[name] = ReplayDataset(
            features=torch.from_numpy(gn.astype(np.float32)),
            gate=torch.from_numpy(stream.gate.astype(np.int64)),
            choice=torch.from_numpy(stream.choice.astype(np.int64)),
            severity=torch.from_numpy(stream.severity.astype(np.int64)),
            stratum=torch.from_numpy(stratum),
            t=torch.from_numpy(np.asarray(stream.t, dtype=np.float64)),
            standardize=stats,
            s_vec=s_scaled,
        )
    return ReplayBundle(datasets=datasets, glass=glass, samples=samples, q_tiles=q_tiles)
