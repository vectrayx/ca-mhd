"""F4 — Campaña del CUL-Bench (plan v3 §3.5–§3.6, §5).

Grid: γ × semilla × (métodos de entrenamiento × políticas de serving).

Métodos de ENTRENAMIENTO del student (ablations del mismo trainer):
- teacher:      el teacher mismo (referencia de calidad, B1)
- B2:           student supervisado (sin KD)
- B3a:          vanilla KD const-α + temp scaling del student (fairness E1)
- B3b:          vanilla KD + oversampling de ejemplos difíciles (ciega a carga)
- LCD:          α(s) + λ_conf + curriculum (método)
- LCD_dagger:   LCD + bucle cerrado DAgger (OURS headline)

Políticas de SERVING (misma traza de test para todos):
- teacher_always            → B1 (techo de calidad)
- student_always            → B2/B3a/B3b/LCD servidos sin routing
- fixed_threshold (sweep)   → B4 (mejor caso sobre val)
- random_matched            → B5 (cobertura igualada al Schmitt)
- schmitt                   → Ours (deadband κ-derivado)

Salida: JSONL por (γ, seed) — una fila por (method, policy). El B9-oráculo
queda como pendiente documentada (DP sobre q discretizada).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .bench.latency import LatencyModel
from .bench.sim import SimConfig, run_des
from .config import RouterConfig, RunConfig, StreamConfig
from .models import ArchConfig
from .dagger import DAggerConfig, run_dagger
from .dataset import ReplayDataset
from .evaluator import StrataTiles, deployed_metrics
from .models import CAMHDNet
from .pipeline import build_replay_bundle
from .policies import (FixedThreshold, RandomMatched, SchmittRouter,
                       StudentAlways, TeacherAlways)
from .smooth_ece import smooth_ece
from .train_lcd import (StudentArtifact, TrainStudentConfig, precompute_targets,
                        train_student)
from .train_teacher import TeacherArtifact, TrainTeacherConfig, train_teacher

__all__ = ["CampaignConfig", "STUDENT_METHODS", "run_cell", "run_campaign",
           "main"]

STUDENT_METHODS = ("B2", "B3a", "B3b", "LCD", "LCD_dagger")


@dataclass
class CampaignConfig:
    gammas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    seeds: tuple[int, ...] = (0, 1, 2)
    horizons: dict = field(default_factory=lambda: {
        "train": 24.0, "val": 8.0, "test": 16.0, "rollout": 8.0})
    arch: ArchConfig = field(default_factory=lambda: ArchConfig(w=128, depth=2))
    teacher_epochs: int = 8
    teacher_batch: int = 512
    teacher_lr: float = 3e-3
    student_lr: float = 2e-3
    student_epochs: int = 4
    student_batch: int = 512
    tau: float = 2.0                       # pre-registro (idéntico en todos)
    lam_conf: float = 0.1
    alpha_base: float = 0.5
    alpha_max: float = 2.0
    smooth_bw: float = 0.05
    sim: SimConfig = field(default_factory=lambda: SimConfig(
        batch_size=16, service_scale=1.0))
    dagger_iters: int = 2
    dagger_oversample: int = 3
    b4_grid: tuple[float, ...] = (0.30, 0.45, 0.60, 0.75, 0.90)
    out_dir: str = "results/campaign"
    torch_threads: int = 1

    def stream_cfg(self, gamma: float) -> StreamConfig:
        # Enmienda del pilot (2026-10-21, PRE-REGISTRATION §8): el punto
        # operativo pre-registrado ρ_t(burst)=1.083 / ρ_s=0.941 se preserva
        # EXACTO (κ/Δserv y deadband son ratios, invariante a escala); el
        # factor artificial ×2.5 del servicio consume el 72% del SLA de 1.5 ms
        # y saturaba viol% ~89% para TODOS los métodos. Física: escala real
        # (service_scale=1.0, tasas ×2.5: 5000/35000, duty 0.25, L_max 1.5 ms).
        return StreamConfig(gamma=gamma, rate_idle=5000.0, rate_burst=35000.0)


def _teacher_cfg(cc: CampaignConfig) -> TrainTeacherConfig:
    return TrainTeacherConfig(arch=cc.arch, epochs=cc.teacher_epochs,
                              batch_size=cc.teacher_batch, lr=cc.teacher_lr,
                              smooth_bw=cc.smooth_bw, log=False)


def _student_cfg(cc: CampaignConfig, method: str) -> TrainStudentConfig:
    common = dict(tau=cc.tau, alpha_base=cc.alpha_base,
                  alpha_max=cc.alpha_max, smooth_bw=cc.smooth_bw, log=False)
    if method == "B2":
        return TrainStudentConfig(use_kd=False, curriculum=False,
                                  lam_conf=0.0, alpha_mode="const", **common)
    if method in ("B3a", "B3b"):
        # plan v3 §baselines: B3a = vanilla KD (post-hoc temp scaling);
        # B3b = vanilla KD + oversampling ciego de errores (se hace en
        # _train_student_method): MISMA pérdida que B3a (α const, λ_conf=0),
        # solo cambia el dataset. El artifact_name evita colisión con B3a.
        return TrainStudentConfig(alpha_mode="const", lam_conf=0.0,
                                  curriculum=False,
                                  artifact_name="B3b_kd_blind_oversample"
                                  if method == "B3b" else None, **common)
    if method == "LCD":
        return TrainStudentConfig(alpha_mode="s", curriculum=True,
                                  lam_conf=cc.lam_conf, **common)
    if method == "LCD_dagger":
        return TrainStudentConfig(alpha_mode="s", curriculum=True,
                                  lam_conf=cc.lam_conf, **common)
    raise ValueError(method)


def _train_student_method(
    cc: CampaignConfig, method: str, teacher: TeacherArtifact,
    bundle, run_cfg: RunConfig, stream_cfg: StreamConfig, lat,
) -> StudentArtifact | None:
    fname = _student_cfg(cc, method)
    if method == "LCD_dagger":
        return run_dagger(
            teacher, bundle.datasets, bundle.samples, bundle.q_tiles, lat,
            cc.sim, stream_cfg, run_cfg, fname,
            DAggerConfig(n_iters=cc.dagger_iters, oversample=cc.dagger_oversample),
            kappa_scale=cc.sim.service_scale if False else 2.5,
            service_scale=cc.sim.service_scale,
            lat_report_path=_LAT[1])
    ds = bundle.datasets
    net = copy.deepcopy(teacher.net)
    with torch.no_grad():
        tgt = precompute_targets(teacher.net, ds["train"].features, ds["train"].s,
                                 teacher.temperatures, fname.tau)
    if method == "B3b":
        # oversampling CIEGO a carga: duplica eventos que el teacher ERRÓ (by
        # argmax de las cabezas) — sin tomar el estado s en cuenta
        with torch.no_grad():
            zT = teacher.net(ds["train"].features, ds["train"].s, mode="teacher")["t"]
            from .losses import corn_pmf
            err = ((zT["gate"].argmax(-1) != ds["train"].gate)
                   | (zT["choice"].argmax(-1) != ds["train"].choice)
                   | (corn_pmf(zT["corn"]).argmax(-1) != ds["train"].severity - 1))
            err = err & (ds["train"].gate == 1)
        idx_err = torch.where(err)[0]
        feats = torch.cat([ds["train"].features, ds["train"].features[idx_err]]) \
            if len(idx_err) else ds["train"].features
        gate = torch.cat([ds["train"].gate, ds["train"].gate[idx_err]]) if len(idx_err) else ds["train"].gate
        choice = torch.cat([ds["train"].choice, ds["train"].choice[idx_err]]) if len(idx_err) else ds["train"].choice
        sev = torch.cat([ds["train"].severity, ds["train"].severity[idx_err]]) if len(idx_err) else ds["train"].severity
        stratum = torch.cat([ds["train"].stratum, ds["train"].stratum[idx_err]]) if len(idx_err) else ds["train"].stratum
        t = torch.cat([ds["train"].t, ds["train"].t[idx_err]]) if len(idx_err) else ds["train"].t
        s = torch.cat([ds["train"].s, ds["train"].s[idx_err]]) if len(idx_err) else ds["train"].s
        aug = ReplayDataset(features=feats, gate=gate, choice=choice, severity=sev,
                            stratum=stratum, t=t, standardize=None, s_vec=s)
        with torch.no_grad():
            tgt = precompute_targets(teacher.net, aug.features, aug.s,
                                     teacher.temperatures, fname.tau)
        ds_eff = {"train": aug, "val": ds["val"], "test": ds["test"]}
    else:
        ds_eff = ds
    return train_student(net, ds_eff, tgt, fname, run_cfg, stream_cfg,
                         teacher.temperatures)


def _deploy(net, policy, sample, ds_labels, lat, sim_cfg, std, tiles, bw):
    res = run_des(sample, net, policy, lat, sim_cfg, std)
    m = deployed_metrics(res, tiles, ds_labels["gate"], ds_labels["choice"],
                         ds_labels["sev"], l_max_us=sim_cfg.l_max_us, bw=bw)
    return m


def run_cell(gamma: float, seed: int, cc: CampaignConfig) -> list[dict]:
    t0 = time.time()
    torch.set_num_threads(cc.torch_threads)
    torch.manual_seed(seed)
    stream_cfg = cc.stream_cfg(gamma)
    lat = LatencyModel.from_json(_LAT[1], batch_size=cc.sim.batch_size)
    run_cfg = RunConfig(train_seed=seed, stream_seed=seed)
    sim_cfg = copy.deepcopy(cc.sim)
    sim_cfg.rate_idle = stream_cfg.rate_idle
    sim_cfg.rate_burst = stream_cfg.rate_burst

    bundle = build_replay_bundle(cc.horizons, stream_cfg, sim_cfg, lat,
                                 seeds={"train": seed, "val": 100 + seed,
                                        "test": 200 + seed, "rollout": 300 + seed})
    run_cfg_teacher = RunConfig(train_seed=seed, stream_seed=seed)
    teacher = train_teacher(bundle.datasets, _teacher_cfg(cc), stream_cfg,
                            run_cfg_teacher)
    mu, sd = bundle.datasets["train"].standardize
    tiles = StrataTiles(*bundle.q_tiles)
    ds_test = bundle.datasets["test"]
    labels = {"gate": ds_test.gate.numpy(), "choice": ds_test.choice.numpy(),
              "sev": ds_test.severity.numpy()}
    y_val = {"gate": bundle.datasets["val"].gate.numpy(),
             "choice": bundle.datasets["val"].choice.numpy(),
             "sev": bundle.datasets["val"].severity.numpy()}
    ds_deploy = copy.deepcopy(sim_cfg)
    ds_deploy.t_end = cc.horizons["test"]

    rows: list[dict] = []

    def _row(method: str, policy_name: str, m: dict) -> None:
        row = {"gamma": gamma, "seed": seed, "method": method,
               "policy": policy_name}
        row.update({k: v for k, v in m["serve"].items()})
        row["smoothECE"] = m["smoothECE"]
        row["accuracy"] = m["accuracy"]
        row["auroc_gate"] = m["gate"].get("auroc")
        rows.append(row)

    # ---- B1 teacher siempre
    mT = _deploy(teacher.net, TeacherAlways(), bundle.samples["test"], labels,
                 lat, ds_deploy, (mu, sd), tiles, cc.smooth_bw)
    _row("teacher", "teacher_always", mT)
    # training-side teacher metrics (B1 quality ref)
    rowT = {"gamma": gamma, "seed": seed, "method": "teacher", "policy": "training",
            "teacher_smoothECE": teacher.metrics["smoothECE"],
            "auroc_gate": teacher.metrics["auroc_gate"],
            "temps": teacher.temperatures}
    rows.append(rowT)

    # ---- student training variants
    nets: dict[str, CAMHDNet] = {}
    for method in STUDENT_METHODS:
        art = _train_student_method(cc, method, teacher, bundle, run_cfg, stream_cfg, lat)
        nets[method] = art.net
        rows.append({"gamma": gamma, "seed": seed, "method": method,
                     "policy": "training",
                     "student_smoothECE": art.metrics["smoothECE"],
                     "student_auroc": art.metrics["auroc_gate"]})
    # ---- serving: student-only (B2/B3a/B3b/LCD servidos de forma fija)
    mS = _deploy(nets["LCD"], StudentAlways(), bundle.samples["test"], labels,
                 lat, ds_deploy, (mu, sd), tiles, cc.smooth_bw)
    _row("LCD", "student_always", mS)
    mS2 = _deploy(nets["B2"], StudentAlways(), bundle.samples["test"], labels,
                  lat, ds_deploy, (mu, sd), tiles, cc.smooth_bw)
    _row("B2", "student_always", mS2)
    mS3 = _deploy(nets["B3a"], StudentAlways(), bundle.samples["test"], labels,
                  lat, ds_deploy, (mu, sd), tiles, cc.smooth_bw)
    _row("B3a", "student_always", mS3)
    mS4 = _deploy(nets["B3b"], StudentAlways(), bundle.samples["test"], labels,
                  lat, ds_deploy, (mu, sd), tiles, cc.smooth_bw)
    _row("B3b", "student_always", mS4)

    # ---- Ours schmitt con nuestro student (HEADLINE)
    import json as _json
    report = _json.loads(open(_LAT[1]).read())
    d = None
    from .policies import derive_thresholds_from_latency
    d = derive_thresholds_from_latency(report,
                                       kappa_scale=cc.sim.service_scale,
                                       service_scale=cc.sim.service_scale)
    rc = RouterConfig(q_on=d["q_on"], q_off=d["q_off"], min_dwell=1.0,
                      latency_refractory=2.0)
    # el router es ESTADOFUL: una instancia fresca por deploy (el autómata
    # no debe arrastrar narrow/guard entre corridas — bug del pilot v1)
    mO = _deploy(nets["LCD_dagger"], copy.deepcopy(SchmittRouter(rc, q_ref=sim_cfg.batch_size)),
                 bundle.samples["test"], labels, lat, ds_deploy, (mu, sd), tiles,
                 cc.smooth_bw)
    _row("LCD_dagger", "schmitt", mO)
    mO2 = _deploy(nets["LCD"], copy.deepcopy(SchmittRouter(rc, q_ref=sim_cfg.batch_size)),
                  bundle.samples["test"], labels, lat, ds_deploy, (mu, sd), tiles,
                  cc.smooth_bw)
    _row("LCD", "schmitt", mO2)
    cov = mO["serve"]["student_coverage"]

    # ---- B5 random con cobertura igualada
    mR = _deploy(nets["LCD"], RandomMatched(cov, seed=seed), bundle.samples["test"],
                 labels, lat, ds_deploy, (mu, sd), tiles, cc.smooth_bw)
    _row("LCD", "random_matched", mR)

    # ---- B4 fixed threshold sweep mejor caso por val
    best = None
    sim_val = copy.deepcopy(sim_cfg)
    sim_val.t_end = cc.horizons["val"]
    for qt in cc.b4_grid:
        mv = _deploy(nets["LCD"], FixedThreshold(q_thresh=qt, q_ref=sim_cfg.batch_size),
                     bundle.samples["val"], y_val, lat, sim_val, (mu, sd), tiles,
                     cc.smooth_bw)
        viol = mv["serve"]["sla_viol_rate"]
        sw = mv["serve"]["n_switches"]
        if best is None or (viol, sw) < (best[1]["serve"]["sla_viol_rate"],
                                         best[1]["serve"]["n_switches"]):
            best = (qt, mv)
    mB4 = _deploy(nets["LCD"], FixedThreshold(q_thresh=best[0], q_ref=sim_cfg.batch_size),
                  bundle.samples["test"], labels, lat, ds_deploy, (mu, sd), tiles,
                  cc.smooth_bw)
    _row("LCD", "fixed_threshold", mB4)
    rows.append({"gamma": gamma, "seed": seed, "method": "B4_choice",
                 "policy": "training", "q_thresh": best[0],
                 "val_viol": best[1]["serve"]["sla_viol_rate"]})

    for r in rows:
        r["wall_s"] = round(time.time() - t0, 1)
    # artefactos para análisis F5 (δ(s), ventanas de transiente, figuras)
    out = Path(cc.out_dir)
    nets_dir = out / "nets"
    nets_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": teacher.net.state_dict(),
                "temperatures": teacher.temperatures,
                "students": {k: v.state_dict() for k, v in nets.items()}},
               nets_dir / f"cell_g{gamma:g}_s{seed}.pt")
    return rows


# estado por proceso (lat report cargado una vez)
_LAT: tuple[LatencyModel | None, str] = (None, "results/latency_report.json")


def _init_worker() -> None:
    from camhd.models import ArchConfig as _AC  # noqa: F401 (warm import)
    torch.set_num_threads(1)


def run_campaign(cfg: CampaignConfig, n_proc: int = 4) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = [(g, s) for g in cfg.gammas for s in cfg.seeds]
    print(f"campaign: {len(cells)} celdas γ×seed, {len(STUDENT_METHODS)+1} métodos")
    jobs = [(g, s, cfg) for (g, s) in cells]
    with mp.get_context("spawn").Pool(n_proc, initializer=_init_worker) as pool:
        for i, rows in enumerate(pool.imap_unordered(_cell_job, jobs)):
            g, s = rows[0]["gamma"], rows[0]["seed"]
            path = out / f"cell_g{g:g}_s{s}.json"
            with open(path, "w") as f:
                json.dump(rows, f, indent=1, default=float)
            print(f"[{i+1}/{len(cells)}] γ={g} seed={s}: {len(rows)} filas")
    print("campaña completa →", out)


def _cell_job(args) -> list[dict]:
    g, s, cfg = args
    return run_cell(g, s, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Campaña CA-MHD F4")
    ap.add_argument("--pilot", action="store_true",
                    help="una celda γ=0.5, seed=0 (validación del harness)")
    ap.add_argument("--out", default="results/campaign")
    ap.add_argument("--gammas", default="0,0.25,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_proc", type=int, default=4)
    a = ap.parse_args()
    cc = CampaignConfig(out_dir=a.out,
                        gammas=tuple(float(x) for x in a.gammas.split(",")),
                        seeds=tuple(int(x) for x in a.seeds.split(",")))
    if a.pilot:
        cc.gammas, cc.seeds = (0.5,), (0,)
    run_campaign(cc, n_proc=a.n_proc)


if __name__ == "__main__":
    main()
