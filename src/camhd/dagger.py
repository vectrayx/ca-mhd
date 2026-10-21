"""Fase C — DAgger condicionado (plan v3 §3.3).

Iteración:
1. ROLLOUT: DES con el student actual + SchmittRouter sobre el split de
   rollout (stream fresco). El s_vec registrado por el DES es el estado REAL
   bajo la política activa (no el del glass box de entrenamiento).
2. EVENTOS DANGER: cabezas servidas en modo student con predicción errada
   (gate/choice/corn). Ground-truth disponible (stream sintético) — DAgger
   legítimo, no requiere anotador.
3. Los danger events se SOBREMUESTREAN (×oversample)ayed al replay train
   con sus s_vec desplegados, y se re-destila. Tras 2–3 iteraciones el
   student corrige exactamente los estados donde el despliegue lo usa mal.

El estado del artefacto final es el último student; cada iteración registra
la métrica desplegada (δ(s) observada) para la curva del paper (Fig. 4).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .bench.sim import SimConfig, run_des
from .config import RouterConfig, RunConfig, StreamConfig
from .dataset import ReplayDataset
from .evaluator import StrataTiles, deployed_metrics
from .models import CAMHDNet
from .policies import SchmittRouter
from .train_lcd import (StudentArtifact, TrainStudentConfig, precompute_targets,
                        train_student)
from .train_teacher import TeacherArtifact

__all__ = ["DAggerConfig", "run_dagger"]


@dataclass
class DAggerConfig:
    n_iters: int = 2
    oversample: int = 3           # réplicas de danger añadidas al replay
    smooth_bw: float = 0.05
    seed_offset: int = 0          # offset del RunConfig por iteración


def _danger_mask(res, sample, labels: dict[str, np.ndarray]) -> np.ndarray:
    n = res.n
    wrong = (res.pred_gate[:n] >= 0.5).astype(np.int64) != labels["gate"][:n]
    wrong |= res.pred_choice[:n] != labels["choice"][:n]
    attack = labels["gate"][:n] == 1
    wrong |= attack & (res.pred_level[:n] != (labels["sev"][:n] - 1))
    # solo eventos SERVIDOS con alguna cabeza en modo student
    return wrong & res.narrow[:n].any(axis=1)


def run_dagger(
    teacher: TeacherArtifact,
    replay: dict[str, ReplayDataset],
    samples: dict,                 # {"train","val","test","rollout"} StreamSample
    glass_q_tiles: tuple[float, float],
    lat,
    sim_cfg: SimConfig,
    stream_cfg: StreamConfig,
    run_cfg: RunConfig,
    tsc: TrainStudentConfig,
    dcfg: DAggerConfig | None = None,
    router_cfg: RouterConfig | None = None,
    kappa_scale: float = 2.5,
    service_scale: float = 2.5,
    lat_report_path: str | None = None,
) -> StudentArtifact:
    """Ejecuta DAgger y devuelve el student final (StudentArtifact)."""
    dcfg = dcfg or DAggerConfig()
    net = copy.deepcopy(teacher.net)
    mu, sd = replay["train"].standardize
    tiles = StrataTiles(*glass_q_tiles)
    rollout_sample = samples["rollout"]

    if router_cfg is None:
        # derivación κ de la campaña; lat_report en results
        import json
        from .policies import derive_thresholds_from_latency
        report = None
        if lat_report_path is not None:
            report = json.loads(open(lat_report_path).read())
        else:
            raise ValueError("DAgger requiere router_cfg o lat_report_path")
        d = derive_thresholds_from_latency(report, kappa_scale=kappa_scale,
                                           service_scale=service_scale)
        router_cfg = RouterConfig(q_on=d["q_on"], q_off=d["q_off"], min_dwell=1.0)

    art: StudentArtifact | None = None
    for it in range(dcfg.n_iters):
        router = SchmittRouter(router_cfg, q_ref=sim_cfg.batch_size,
                               use_latency_guard=True)
        res = run_des(rollout_sample, net, router, lat, sim_cfg, (mu, sd))
        n = res.n
        labels = {"gate": rollout_sample.gate[:n], "choice": rollout_sample.choice[:n],
                  "sev": rollout_sample.severity[:n]}
        danger = _danger_mask(res, rollout_sample, labels)
        n_danger = int(danger.sum())
        # ---- replay aumentado con los danger del rollout (s desplegado)
        feats_r = (rollout_sample.features[:n] - mu) / np.maximum(sd, 1e-6)
        rows = {
            "features": torch.from_numpy(np.concatenate(
                [feats_r] + [feats_r[danger]] * dcfg.oversample).astype(np.float32)),
            "gate": torch.from_numpy(np.concatenate(
                [rollout_sample.gate[:n]] + [labels["gate"][danger]] * dcfg.oversample).astype(np.int64)),
            "choice": torch.from_numpy(np.concatenate(
                [rollout_sample.choice[:n]] + [labels["choice"][danger]] * dcfg.oversample).astype(np.int64)),
            "sev": torch.from_numpy(np.concatenate(
                [rollout_sample.severity[:n]] + [labels["sev"][danger]] * dcfg.oversample).astype(np.int64)),
            "s_vec": torch.from_numpy(np.concatenate(
                [res.s_vec[:n]] + [res.s_vec[:n][danger]] * dcfg.oversample).astype(np.float32)),
        }
        # estrato desplegado de las filas (tiles del train glass box)
        strata_tr = tiles.of(replay["train"].s_vec[:, 0].numpy() if
                             replay["train"].s_vec is not None else np.zeros(len(replay["train"])))
        strata_r = tiles.of(res.s_vec[:n, 0])
        row_strata = np.concatenate(
            [strata_r] + [strata_r[danger]] * dcfg.oversample).astype(np.int64)
        n_train = len(replay["train"])
        t_all = torch.cat([replay["train"].t,
                           torch.from_numpy(np.asarray(rollout_sample.t[:n], dtype=np.float64))])
        aug = ReplayDataset(
            features=torch.cat([replay["train"].features, rows["features"]]),
            gate=torch.cat([replay["train"].gate, rows["gate"]]),
            choice=torch.cat([replay["train"].choice, rows["choice"]]),
            severity=torch.cat([replay["train"].severity, rows["sev"]]),
            stratum=torch.cat([replay["train"].stratum,
                               torch.from_numpy(row_strata)]),
            t=t_all,
            standardize=None,   # features YA estandarizadas (train stats)
            s_vec=torch.cat([replay["train"].s, rows["s_vec"]
                             / torch.as_tensor(sim_cfg.s_scale, dtype=torch.float32)]),
        )
        # ---- re-destilación con el replay aumentado
        with torch.no_grad():
            tgt = precompute_targets(teacher.net, aug.features, aug.s,
                                     teacher.temperatures, tsc.tau)
        run_it = RunConfig(train_seed=run_cfg.train_seed + dcfg.seed_offset + it,
                           stream_seed=run_cfg.stream_seed)
        art = train_student(net, {"train": aug, "val": replay["val"],
                                  "test": replay["test"]},
                            tgt, tsc, run_it, stream_cfg, teacher.temperatures)
        net = art.net
        art.metrics["dagger_iter"] = it
        art.metrics["danger_n"] = n_danger

    assert art is not None
    return art
