"""Experimento de calibración de temperaturas (Enmienda 2).

Hipótesis: fit_temperatures minimiza la NLL en val; el criterio G1 mide
smoothECE. Re-ajustar T por cabeza a minimizar el smoothECE de val puede
reducir el smoothECE de test del teacher bajo 1% sin retocar el training.

Uso (requiere nets guardados por la campaña):
  PYTHONPATH=src .venv/bin/python scripts/exp_calibration.py

Salida: smoothECE por cabeza×estrato en TEST para (a) T_NLL ( las corridas),
(b) T_smoothECE (experimento). NO modifica nada salvo el reporte; la
adopción se decide con la enmienda correspondiente.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from camhd.config import RunConfig, StreamConfig
from camhd.calibration import fit_temperatures
from camhd.dataset import ReplayDataset
from camhd.models import CAMHDNet, ArchConfig
from camhd.pipeline import build_replay_bundle
from camhd.bench.sim import SimConfig
from camhd.bench.latency import LatencyModel
from camhd.train_teacher import smooth_ece_by_head_stratum
from camhd.generator import CoupledStream


def load_bundle_val_test(horizons: dict[str, float], gamma: float = 0.5):
    stream_cfg = StreamConfig(gamma=gamma, rate_idle=5000.0, rate_burst=35000.0)
    lat = LatencyModel.from_json("results/latency_report.json", batch_size=16)
    sim_cfg = SimConfig(batch_size=16, service_scale=1.0)
    sim_cfg.rate_idle = stream_cfg.rate_idle
    sim_cfg.rate_burst = stream_cfg.rate_burst
    bundle = build_replay_bundle(horizons, stream_cfg, sim_cfg,
                                 lat, seeds={"train": 0, "val": 100, "test": 200})
    return bundle


def main() -> None:
    run_cfg = RunConfig(train_seed=0, stream_seed=0)
    horizons = {"train": 24.0, "val": 8.0, "test": 16.0, "rollout": 8.0}
    bundle = load_bundle_val_test(horizons)
    state = torch.load("results/campaign/nets/cell_g0.5_s0.pt",
                       map_location="cpu", weights_only=False)
    temps_old = state["temperatures"]
    net = CAMHDNet(ArchConfig(w=128, depth=2))
    net.load_state_dict(state["teacher"])
    net.eval()
    val = bundle.datasets["val"]
    test = bundle.datasets["test"]

    with torch.no_grad():
        lv = net(val.features, val.s, mode="teacher")["t"]
    # attack mask para corn
    val_attack = val.gate == 1
    logits = {"gate": lv["gate"], "choice": lv["choice"],
              "corn": lv["corn"][val_attack]}
    targets = {"gate": val.gate, "choice": val.choice,
               "corn": val.severity[val_attack] - 1}

    temps_nll = fit_temperatures(logits, targets)
    temps_se = fit_temperatures(logits, targets, objective="smooth_ece")
    print("T_NLL      :", {k: round(v, 4) for k, v in temps_nll.items()})
    print("T_smoothECE:", {k: round(v, 4) for k, v in temps_se.items()})
    print("T_nll (guardadas):", {k: round(v, 4) for k, v in temps_old.items()})

    se_old = smooth_ece_by_head_stratum(net, test, temps_old, bw=0.05)
    se_nll = smooth_ece_by_head_stratum(net, test, temps_nll, bw=0.05)
    se_new = smooth_ece_by_head_stratum(net, test, temps_se, bw=0.05)
    for name, se in (("guardadas", se_old), ("fit NLL", se_nll),
                     ("fit smoothECE", se_new)):
        print(f"--- teacher TEST smoothECE ({name}) ---")
        for s in ("all", "calm", "warm", "hot"):
            print(f"  {s:5s}: " + "  ".join(
                f"{h}={se[s][h]*100:5.2f}%" for h in ("gate", "choice", "corn")))


if __name__ == "__main__":
    main()
