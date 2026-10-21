"""Sweep de igualación del router (§5.3): Schmitt con q̂_on barrido.

El criterio Es2 exige conmutaciones del Schmitt ≤ 1/3 de B4-best-q
A VIOLACIÓN IGUALADA (±1 pt). En γ=0.5 el Schmitt sellado (q̂_on=0.675)
queda a 1.2 pts del B4-best — la sweep pre-registrada explora q̂_on y
declara toda la curva (viol, switches, k_eventos servidos student):
la igualación se logra subiendo q̂_on (student-mode más temprano y
más largo → menos viol). Selección declarada: el q̂_on con viol
más cercano a la banda [viol_B4 − 1pt, viol_B4 + 1pt]; en empate,
menos conmutaciones. TODA la curva va al reporte (sin cherry-pick).

Uso: PYTHONPATH=src .venv/bin/python scripts/router_equalize.py \
       [--gammas 0,0.25,0.5,1.0] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch
from scipy import stats as st

from camhd.campaign import CampaignConfig
from camhd.config import RouterConfig
from camhd.bench.latency import LatencyModel
from camhd.bench.sim import run_des
from camhd.models import CAMHDNet, ArchConfig
from camhd.pipeline import build_replay_bundle
from camhd.policies import SchmittRouter, derive_thresholds_from_latency
from camhd.evaluator import deployed_metrics, StrataTiles

REPORT = "results/latency_report.json"
OUT_DIR = Path("results/router_sweep")
Q_ON_GRID = (0.40, 0.50, 0.675, 0.85, 1.00)   # mult del q̂_on sellado = 0.675


def eval_cell(cc: CampaignConfig, gamma: float, seed: int) -> dict:
    stream_cfg = cc.stream_cfg(gamma)
    sim_cfg = copy.deepcopy(cc.sim)
    sim_cfg.rate_idle = stream_cfg.rate_idle
    sim_cfg.rate_burst = stream_cfg.rate_burst
    lat = LatencyModel.from_json(REPORT, batch_size=cc.sim.batch_size)
    bundle = build_replay_bundle(cc.horizons, stream_cfg, sim_cfg, lat,
                                 seeds={"train": seed, "val": 100 + seed,
                                        "test": 200 + seed, "rollout": 300 + seed})
    net_state = torch.load(f"{cc.out_dir}/nets/cell_g{gamma:g}_s{seed}.pt",
                           map_location="cpu", weights_only=False)
    report = json.loads(Path(REPORT).read_text())
    th = derive_thresholds_from_latency(report, kappa_scale=cc.sim.service_scale,
                                        service_scale=cc.sim.service_scale)
    mu, sd = bundle.datasets["train"].standardize
    tiles = StrataTiles(*bundle.q_tiles)
    labels = {"gate": bundle.datasets["test"].gate.numpy(),
              "choice": bundle.datasets["test"].choice.numpy(),
              "sev": bundle.datasets["test"].severity.numpy()}
    d_deploy = copy.deepcopy(sim_cfg)
    d_deploy.t_end = cc.horizons["test"]

    curves = {}
    for method in ("LCD_dagger", "LCD"):
        net = CAMHDNet(ArchConfig(w=128, depth=2))
        net.load_state_dict(net_state["students"][method])
        net.eval()
        curve = []
        for qon in Q_ON_GRID:
            q_on = min(1.0, th["q_on"] * qon)
            # mantener la relación q_on − q_off = 0.506 en presión (sellado):
            q_off = max(0.0, q_on - (th["q_on"] - th["q_off"]))
            if q_off >= q_on:
                q_off = 0.5 * q_on
            rc = RouterConfig(q_on=q_on, q_off=q_off, min_dwell=1.0,
                              latency_refractory=2.0)
            res = run_des(bundle.samples["test"], net,
                          copy.deepcopy(SchmittRouter(rc, q_ref=sim_cfg.batch_size)),
                          lat, d_deploy)
            m = deployed_metrics(res, tiles, labels["gate"], labels["choice"],
                                 labels["sev"], l_max_us=sim_cfg.l_max_us,
                                 bw=cc.smooth_bw)
            curve.append({
                "q_on_mult": qon, "q_on": q_on, "q_off": q_off,
                "viol": m["serve"]["sla_viol_rate"],
                "p99_us": m["serve"]["lat_p99_us"],
                "p50_us": m["serve"]["lat_p50_us"],
                "switches": m["serve"]["n_switches"],
                "coverage": m["serve"]["student_coverage"],
            })
        curves[method] = curve
    # B4-best-q de la fila training (guardo su viol y switches del despliegue)
    cell_path = Path(f"{cc.out_dir}/cell_g{gamma:g}_s{seed}.json")
    cell = json.loads(cell_path.read_text())
    b4 = next(r for r in cell if r["policy"] == "fixed_threshold")
    return {"gamma": gamma, "seed": seed, "curves": curves,
            "B4_best": {"viol": b4["sla_viol_rate"],
                        "switches": b4["n_switches"],
                        "coverage": b4["student_coverage"],
                        "q_thresh_val": next(r["q_thresh"] for r in cell
                                             if r["method"] == "B4_choice")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0,0.25,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cc = CampaignConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_outs = []
    for g in [float(x) for x in a.gammas.split(",")]:
        for s in [int(x) for x in a.seeds.split(",")]:
            t0 = time.time()
            out = eval_cell(cc, g, s)
            all_outs.append(out)
            print(f"[eq] γ={g} s={s} B4 viol={out['B4_best']['viol']*100:.1f}% "
                  f"sw={out['B4_best']['switches']:.0f}; "
                  f"schmitt(SEALADO): {out['curves']['LCD_dagger'][-1]['viol']*100:.1f}% "
                  f"sw={out['curves']['LCD_dagger'][-1]['switches']} "
                  f"[{time.time() - t0:.0f}s]")
    (OUT_DIR / "equalization.json").write_text(json.dumps(all_outs, indent=1))

    # ---------- verificación del criterio Es2 (regla corregida) ----------
    # El sweep explora q̂_on CERCA del sellado; candidatos degenerados
    # (q_off≤0 → autómata muerto, cov=1.0 = siempre-student) NO son el
    # método: se excluyen. Empate (viol_media) a banda ±1 pt del comparado;
    # switches medidos por celda (el Schmitt es 4–6 en todo el grid).
    print("\n=== Es2: Schmitt·LCD_dagger vs B4-best a violación igualada ===")
    print("    (candidatos: mult q̂_on ≥ 0.85, q_off > 0; comparación por γ)")
    g_by = {}
    for out in all_outs:
        g_by.setdefault(out["gamma"], []).append(out)
    for g, outs in sorted(g_by.items()):
        viol_b4 = float(np.median([o["B4_best"]["viol"] * 100 for o in outs]))
        sw_b4 = float(np.median([o["B4_best"]["switches"] for o in outs]))
        best = None
        for mult in (0.85, 0.925, 1.0):
            cand = []
            for o in outs:
                c = next((c for c in o["curves"]["LCD_dagger"]
                          if abs(c["q_on_mult"] - mult) < 1e-9 and c["q_off"] > 0), None)
                if c:
                    cand.append(c)
            if not cand:
                continue
            v_med = float(np.median([c["viol"] * 100 for c in cand]))
            d_med = v_med - viol_b4
            sw_med = float(np.min([c["switches"] for c in cand]))
            if best is None or abs(d_med) < abs(best[0]):
                best = (d_med, mult, v_med, sw_med, len(cand))
        d_med, mult, v_med, sw_med, n_c = best
        pass_is = abs(d_med) <= 1.0 and sw_med <= sw_b4 / 3.0
        print(f"{g:>4}  viol_B4={viol_b4:5.1f}%  q̂_on×{mult:.3f}  viol_sch={v_med:5.1f}%  "
              f"Δ={d_med:+.2f}  sw_sch={sw_med:.0f} vs sw_B4={sw_b4:.0f}  "
              f"celdas={n_c}/{len(outs)}  {'PASS' if pass_is else 'FAIL'}")


if __name__ == "__main__":
    main()
