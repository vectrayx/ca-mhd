"""Ablations G3 (§6 pre-registro) sobre teacher congelado guardado.

Re-entrena STUDENTS reutilizando los teachers de results/campaign/nets/
(sin re-entrenar teachers):
  - LCD_lc0    : λ_conf = 0 (sin término de confianza teacher)
  - LCD_nc0    : curriculum OFF (α(s) desde la época 0)
  - DAgger_noOv: DAgger con oversample=1
Evalúa smoothECE×estrato (protocolo glass-box) y guarda
results/ablations/cell_gG_sS.json con la misma estructura de fila.

Uso: PYTHONPATH=src .venv/bin/python -m scripts.ablations \
        [--gammas 0,0.25,0.5,1.0] [--seeds 0,1,2] [--n_proc 6]
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import torch

from camhd.campaign import CampaignConfig, _teacher_cfg
from camhd.config import RouterConfig, RunConfig
from camhd.bench.latency import LatencyModel
from camhd.bench.sim import SimConfig, run_des
from camhd.models import ArchConfig, CAMHDNet
from camhd.pipeline import build_replay_bundle
from camhd.dagger import DAggerConfig, run_dagger
from camhd.evaluator import deployed_metrics, StrataTiles
from camhd.policies import FixedThreshold, RandomMatched, SchmittRouter, \
    derive_thresholds_from_latency
from camhd.train_lcd import (StudentArtifact, TrainStudentConfig,
                             precompute_targets, train_student)
from camhd.train_teacher import TeacherArtifact

_REPORT = "results/latency_report.json"


def _load_teacher(state: dict, arch: ArchConfig) -> TeacherArtifact:
    net = CAMHDNet(arch)
    net.load_state_dict(state["teacher"])
    net.eval()
    return TeacherArtifact(net=net, arch=arch, stream=None, run=None,
                           temperatures=state["temperatures"],
                           kendall_lv={}, metrics={}, standardize=None)


def _student_cfg(name: str, cc: CampaignConfig) -> TrainStudentConfig:
    common = dict(tau=cc.tau, alpha_base=cc.alpha_base, alpha_max=cc.alpha_max,
                  smooth_bw=cc.smooth_bw, log=False)
    if name == "LCD_lc0":
        return TrainStudentConfig(alpha_mode="s", curriculum=True,
                                  lam_conf=0.0, artifact_name="LCD_lc0", **common)
    if name == "LCD_nc0":
        return TrainStudentConfig(alpha_mode="s", curriculum=False,
                                  lam_conf=cc.lam_conf,
                                  artifact_name="LCD_nc0", **common)
    if name == "DAgger_noOv":
        return TrainStudentConfig(alpha_mode="s", curriculum=True,
                                  lam_conf=cc.lam_conf,
                                  artifact_name="DAgger_noOv", **common)
    raise ValueError(name)


def _deploy_once(net, policy, sample, labels, lat, sim_cfg, mu_sd, tiles, bw):
    res = run_des(sample, net, policy, lat, sim_cfg, mu_sd)
    return deployed_metrics(res, tiles, labels["gate"], labels["choice"],
                            labels["sev"], l_max_us=sim_cfg.l_max_us, bw=bw)


def _cell(cc: CampaignConfig, gamma: float, seed: int) -> list[dict]:
    t0 = time.time()
    torch.set_num_threads(cc.torch_threads)
    torch.manual_seed(seed)
    stream_cfg = cc.stream_cfg(gamma)
    sim_cfg = copy.deepcopy(cc.sim)
    sim_cfg.rate_idle = stream_cfg.rate_idle
    sim_cfg.rate_burst = stream_cfg.rate_burst
    lat = LatencyModel.from_json(_REPORT, batch_size=cc.sim.batch_size)
    bundle = build_replay_bundle(cc.horizons, stream_cfg, sim_cfg, lat,
                                 seeds={"train": seed, "val": 100 + seed,
                                        "test": 200 + seed, "rollout": 300 + seed})
    run_cfg = RunConfig(train_seed=seed, stream_seed=seed)
    state = torch.load(f"{cc.out_dir}/nets/cell_g{gamma:g}_s{seed}.pt",
                       map_location="cpu", weights_only=False)
    teacher = _load_teacher(state, cc.arch)
    tiles = StrataTiles(*bundle.q_tiles)
    labels = {"gate": bundle.datasets["test"].gate.numpy(),
              "choice": bundle.datasets["test"].choice.numpy(),
              "sev": bundle.datasets["test"].severity.numpy()}
    mu, sd = bundle.datasets["train"].standardize
    d_deploy = copy.deepcopy(sim_cfg)
    d_deploy.t_end = cc.horizons["test"]

    rows = []
    for name in ("LCD_lc0", "LCD_nc0", "DAgger_noOv"):
        tsc = _student_cfg(name, cc)
        if name == "DAgger_noOv":
            art = run_dagger(
                teacher, bundle.datasets, bundle.samples, bundle.q_tiles, lat,
                sim_cfg, stream_cfg, run_cfg, tsc,
                DAggerConfig(n_iters=cc.dagger_iters, oversample=1),
                kappa_scale=cc.sim.service_scale,
                service_scale=cc.sim.service_scale, lat_report_path=_REPORT)
        else:
            net = copy.deepcopy(teacher.net)
            with torch.no_grad():
                tgt = precompute_targets(teacher.net, bundle.datasets["train"].features,
                                         bundle.datasets["train"].s,
                                         teacher.temperatures, tsc.tau)
            art = train_student(net, bundle.datasets, tgt, tsc, run_cfg,
                                stream_cfg, teacher.temperatures)
        # deploy Schmitt sellado (comparable con el grid)
        report = json.loads(Path(_REPORT).read_text())
        th = derive_thresholds_from_latency(report,
                                           kappa_scale=cc.sim.service_scale,
                                           service_scale=cc.sim.service_scale)
        rc = RouterConfig(q_on=th["q_on"], q_off=th["q_off"], min_dwell=1.0,
                          latency_refractory=2.0)
        m = _deploy_once(art.net, copy.deepcopy(SchmittRouter(rc, q_ref=sim_cfg.batch_size)),
                         bundle.samples["test"], labels, lat, d_deploy,
                         (mu, sd), tiles, cc.smooth_bw)
        rows.append({"gamma": gamma, "seed": seed, "method": name,
                     "variant": tsc.variant_name(),
                     "student_smoothECE": art.metrics["smoothECE"],
                     "auroc_gate": art.metrics["auroc_gate"],
                     "serve": m["serve"],
                     "wall_s": round(time.time() - t0, 1)})
    return rows


def worker(args: tuple[float, int]) -> list[dict]:
    gamma, seed = args
    cc = CampaignConfig()
    return _cell(cc, gamma, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0,0.25,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_proc", type=int, default=6)
    a = ap.parse_args()
    jobs = [(float(g), int(s))
            for g in a.gammas.split(",") for s in a.seeds.split(",")]
    out_dir = Path("results/ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.Pool(a.n_proc) as pool:
        for i, rows in enumerate(pool.imap_unordered(worker, jobs)):
            rows[0].setdefault("meta", {})
            rows[0]["meta"]["wall_s"] = round(time.time() - t0, 1)
            g, s = rows[0]["gamma"], rows[0]["seed"]
            (out_dir / f"cell_g{g:g}_s{s}.json").write_text(json.dumps(rows, indent=1))
            print(f"[abl {i+1}/{len(jobs)}] γ={g} s={s}: {len(rows)} filas "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"ablations completas → {out_dir}")


if __name__ == "__main__":
    main()
