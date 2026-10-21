"""F5: análisis de transiente ECE-swap alrededor de conmutaciones.

PRE: la parrilla ya guardó nets por celda (results/campaign/nets/cell_*.pt).

Para cada celda (γ, seed) especificada:
1. Reconstruye el bundle EXACTO de la campaña (mismas seeds y tasas).
2. Re-ejecuta el despliegue Schmitt del LCD_dagger (la política headline)
   y vuelca por-evento: t_arr, narrow (por cabeza), lat_us, q_s, s_vec,
   y el correctness/confidence del gate/choice/corn servidos.
3. Del trace: ventanas ±2 s alrededor de cada conmutación (t_salvado por
   el DES en el instante del flip); n_min=200 por ventana; bootstrapea
   jerárquico (resample por periodo de burst) el smoothECE por ventana.
4. Salida: results/f5/cell_gG_sS.json con:
   - switches: [{t, q, frm, to}]
   - por ventana (antes/después): smoothECE gate/choice/corn, n, IC boot.

Uso:  PYTHONPATH=src .venv/bin/python scripts/f5_transient.py [--gammas 0.5 --seeds 0]
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

from camhd.campaign import CampaignConfig
from camhd.config import RouterConfig
from camhd.bench.latency import LatencyModel
from camhd.bench.sim import run_des
from camhd.models import CAMHDNet, ArchConfig
from camhd.pipeline import build_replay_bundle
from camhd.policies import SchmittRouter, derive_thresholds_from_latency
from camhd.evaluator import deployed_metrics, StrataTiles
from camhd.smooth_ece import smooth_ece

REPORT = "results/latency_report.json"
OUT_DIR = Path("results/f5")
WINDOW = 2.0          # ±2 s alrededor del switch (§5 E2)
N_MIN = 200           # n mínimo por ventana
N_BOOT = 200          # bloques resampleados (IC 90%); predicción puntual exacta


def boot_stratified_se(conf: np.ndarray, corr: np.ndarray, bw: float,
                       n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Bootstrap de grupos (periodos de burst ≈ trozos de 8 s) para el
    smoothECE. Superior jerárquico: resample SECUENCIAL de bloques no
    solapados de duración 1 s (para conservar la correlación temporal)."""
    n = conf.size
    if n < N_MIN:
        return float("nan"), float("nan"), float("nan")
    edges = np.arange(0, n + 1, max(1, int(n / 16)))  # 16 bloques
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([blocks[i] for i in
                               np.random.randint(0, len(blocks), size=len(blocks))])
        stats[b] = smooth_ece(conf[pick], corr[pick], bandwidth=bw)
    return float(smooth_ece(conf, corr, bandwidth=bw)), \
        float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def trace_cell(cc: CampaignConfig, gamma: float, seed: int, bw: float) -> dict:
    stream_cfg = cc.stream_cfg(gamma)
    sim_cfg = copy.deepcopy(cc.sim)
    sim_cfg.rate_idle = stream_cfg.rate_idle
    sim_cfg.rate_burst = stream_cfg.rate_burst
    lat = LatencyModel.from_json(REPORT, batch_size=cc.sim.batch_size)
    bundle = build_replay_bundle(cc.horizons, stream_cfg, sim_cfg, lat,
                                 seeds={"train": seed, "val": 100 + seed,
                                        "test": 200 + seed, "rollout": 300 + seed})
    net_state = torch.load(cc.out_dir + f"/nets/cell_g{gamma:g}_s{seed}.pt",
                           map_location="cpu", weights_only=False)
    net = CAMHDNet(ArchConfig(w=128, depth=2))
    net.load_state_dict(net_state["students"]["LCD_dagger"])
    net.eval()

    report = json.loads(Path(REPORT).read_text())
    th = derive_thresholds_from_latency(report, kappa_scale=cc.sim.service_scale,
                                        service_scale=cc.sim.service_scale)
    rc = RouterConfig(q_on=th["q_on"], q_off=th["q_off"], min_dwell=1.0,
                      latency_refractory=2.0)
    d_deploy = copy.deepcopy(sim_cfg)
    d_deploy.t_end = cc.horizons["test"]

    mu, sd = bundle.datasets["train"].standardize
    res = run_des(bundle.samples["test"], net,
                  copy.deepcopy(SchmittRouter(rc, q_ref=sim_cfg.batch_size)),
                  lat, d_deploy, (mu, sd))

    # reconstruir de dónde vinieron las predicciones del DES para evaluar
    labels = {"gate": bundle.datasets["test"].gate.numpy(),
              "choice": bundle.datasets["test"].choice.numpy(),
              "sev": bundle.datasets["test"].severity.numpy()}
    m = deployed_metrics(res, StrataTiles(*bundle.q_tiles), labels["gate"],
                         labels["choice"], labels["sev"],
                         l_max_us=sim_cfg.l_max_us, bw=bw)

    # conmutaciones = eventos con κ (=1); DENTRO de un switch el batch entero
    # lleva κ (n≡1 por evento) → agrupar instantes: un switch real = cluster
    # de eventos κ con gaps pequeños.
    idx_k = np.where(np.asarray(res.kappa_n[:res.n]) == 1)[0]
    sw_t_all = np.asarray(res.t_arr)[idx_k]
    sw_q_all = np.asarray(res.q_s)[idx_k]
    n_before = len(sw_t_all)
    keep = np.concatenate(([True], np.diff(sw_t_all) > 0.25))
    sw_t = sw_t_all[keep]
    sw_q = sw_q_all[keep]
    print(f"[f5] κ-eventos={n_before} → switches dedup={len(sw_t)}")
    # correct de gate servido (predicciones del DES, mismo criterio evaluator)
    correct_g = ((res.pred_gate[:res.n] >= 0.5).astype(np.int64)
                 == labels["gate"][:res.n]).astype(float)
    conf_g = np.maximum(res.pred_gate[:res.n], 1.0 - res.pred_gate[:res.n])
    t_arr = np.asarray(bundle.samples["test"].t)[:res.n]
    correct_c = (res.pred_choice[:res.n] == labels["choice"][:res.n]).astype(float)
    conf_c = res.conf_choice[:res.n]
    m_corn = labels["gate"][:res.n] == 1
    corr_corn = np.zeros(res.n)
    corr_corn[m_corn] = (res.pred_level[:res.n][m_corn] ==
                         labels["sev"][:res.n][m_corn] - 1)
    conf_corn = res.conf_corn[:res.n]

    windows = []
    for k, t0s in enumerate(sw_t):
        near = np.abs(t_arr - t0s) <= WINDOW
        pre_m = near & (t_arr <= t0s)
        post_m = near & (t_arr > t0s)
        row = {"switch": k, "t": float(t0s), "q": float(sw_q[k]),
               "n_pre": int(pre_m.sum()), "n_post": int(post_m.sum())}
        for name, (c_, y_, extra) in (
                ("gate", (conf_g, correct_g, np.ones(res.n, bool))),
                ("choice", (conf_c, correct_c, np.ones(res.n, bool))),
                ("corn", (conf_corn[:res.n], corr_corn,
                          m_corn[:res.n]))):
            sel_pre = pre_m & extra
            sel_post = post_m & extra
            pre_v = boot_stratified_se(c_[sel_pre], y_[sel_pre], bw)
            post_v = boot_stratified_se(c_[sel_post], y_[sel_post], bw)
            row[name] = {"pre": {"n": int(sel_pre.sum()), "se": pre_v[0],
                                 "ci": [pre_v[1], pre_v[2]]},
                         "post": {"n": int(sel_post.sum()), "se": post_v[0],
                                  "ci": [post_v[1], post_v[2]]}}
        windows.append(row)
    return {"gamma": gamma, "seed": seed,
            "serve": m["serve"], "switches": len(sw_t), "windows": windows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0.5")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cc = CampaignConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for g in [float(x) for x in a.gammas.split(",")]:
        for s in [int(x) for x in a.seeds.split(",")]:
            t0 = time.time()
            out = trace_cell(cc, g, s, bw=cc.smooth_bw)
            p = OUT_DIR / f"cell_g{g:g}_s{s}.json"
            p.write_text(json.dumps(out, indent=1))
            print(f"[f5] γ={g} s={s}: {out['switches']} switches en "
                  f"{time.time() - t0:.0f}s → {p}")


if __name__ == "__main__":
    main()
