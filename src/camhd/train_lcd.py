"""Phase B — Student training by distillation (plan v3 §3.3).

LCD = L = Σ_q kendall(L_q; σ_q) + α(s)·KL_τ(T‖S) + λ_conf·L_conf-cal

- τ (pre-registered) applied to BOTH sides (also on the student) — the
same temperature for all methods.
- CORN: soft-target ordinal with the pmf of the teacher tempered.
- The baselines of the plan are ABLATIONS of the same trainer (hygiene):
use_kd=False → B2 student-only (narrow supervised)
alpha_mode="const" → B3a vanilla KD (state importance ignored)
alpha_mode="s" → LCD (method: load conditional importance)
λ_conf>0 with α const → B3b KD + calibration term
- Gradient flows ONLY through student-heads + trunk (G3): teacher frozen.
- Teacher-targets precomputed once per teacher (with no_grad).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .losses import corn_loss, corn_pmf, kendall_loss
from .models import CAMHDNet, ArchConfig
from .config import RunConfig, StreamConfig
from .dataset import ReplayDataset
from .smooth_ece import smooth_ece

__all__ = ["TrainStudentConfig", "StudentArtifact", "precompute_targets",
           "train_student", "load_student_artifact"]

_HN = ("gate", "choice", "corn")


@dataclass
class TrainStudentConfig:
    tau: float = 2.0                 # pre-registration (identical in all methods)
    lam_conf: float = 0.1
    alpha_base: float = 0.5
    alpha_max: float = 2.0
    alpha_mode: str = "s"            # "s" (LCD) | "const" (vanilla KD) | "none"
    curriculum: bool = True          # epoch 0 with global mixing (Phase B)
    use_kd: bool = True              # False = B2 (student supervisado)
    epochs: int = 4
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-2
    smooth_bw: float = 0.05
    device: str = "cpu"
    log: bool = True
    artifact_name: str | None = None   # artifact name override (B3b)

    def variant_name(self) -> str:
        if self.artifact_name is not None:
            return self.artifact_name
        if not self.use_kd:
            return "B2_student_only"
        if self.alpha_mode == "const" and self.lam_conf == 0.0:
            return "B3a_vanilla_kd"
        if self.alpha_mode == "const":
            return "B3b_kd_confocal"
        if self.alpha_mode == "s":
            return f"LCD_amax{self.alpha_max:g}_lconf{self.lam_conf:g}"
        return "custom"


@dataclass
class StudentArtifact:
    net: CAMHDNet
    arch: ArchConfig
    temperatures: dict[str, float]      # del STUDENT (fit sobre val post-train)
    kendall_lv: dict[str, float]
    config: TrainStudentConfig
    metrics: dict
    teacher_temps: dict[str, float]     # teacher's kindpl used in distillation

    def save(self, path: str | Path) -> None:
        torch.save({
            "net_state": self.net.state_dict(),
            "arch": asdict(self.arch),
            "temperatures": self.temperatures,
            "kendall_lv": self.kendall_lv,
            "config": asdict(self.config),
            "metrics": self.metrics,
            "teacher_temps": self.teacher_temps,
            "format": "camhd-student-v1",
        }, str(path))


def load_student_artifact(path: str | Path) -> StudentArtifact:
    blob = torch.load(str(path), weights_only=False)
    assert blob.get("format") == "camhd-student-v1", blob.get("format")
    arch = ArchConfig(**blob["arch"])
    net = CAMHDNet(arch)
    net.load_state_dict(blob["net_state"])
    cfg = TrainStudentConfig(**blob["config"])
    return StudentArtifact(net=net, arch=arch, temperatures=blob["temperatures"],
                           kendall_lv=blob["kendall_lv"], config=cfg,
                           metrics=blob["metrics"], teacher_temps=blob["teacher_temps"])


@torch.no_grad()
def precompute_targets(net: CAMHDNet, xs: torch.Tensor, ss: torch.Tensor,
                       temps: dict[str, float], tau: float) -> dict[str, torch.Tensor]:
    """Teacher soft-targets, scaled space, tempered to τ. Batch-efficient."""
    net.eval()
    N = xs.size(0)
    outs: dict[str, torch.Tensor] = {
        "gate": torch.empty(N, 2, dtype=torch.float32),
        "choice": torch.empty(N, net.cfg.n_choice, dtype=torch.float32),
        "corn": torch.empty(N, net.cfg.n_sev, dtype=torch.float32),
    }
    with torch.no_grad():
        for a, b in [(i, min(i + 65536, N)) for i in range(0, N, 65536)]:
            z = net(xs[a:b], ss[a:b], mode="teacher")["t"]
            outs["gate"][a:b] = F.softmax(z["gate"] / tau, dim=-1)
            outs["choice"][a:b] = F.softmax(z["choice"] / tau, dim=-1)
            outs["corn"][a:b] = corn_pmf(z["corn"] / tau)
    return outs


def smooth_ece_strata(net: CAMHDNet, ds: ReplayDataset, temps,
                      bw: float, mode: str = "student") -> dict[str, dict[str, float]]:
    """smoothECE per head × stratum of the given mode (student or teacher)."""
    net.eval()
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        z = net(ds.features, ds.s, mode=mode)
    side = "s" if mode == "student" else "t"
    probs_gate = F.softmax(z[side]["gate"] / temps["gate"], -1)
    probs_choice = F.softmax(z[side]["choice"] / temps["choice"], -1)
    pmf_corn = corn_pmf(z[side]["corn"] / temps["corn"])
    heads = {
        "gate": (probs_gate.max(-1).values,
                 (probs_gate.argmax(-1) == ds.gate).float()),
        "choice": (probs_choice.max(-1).values,
                   (probs_choice.argmax(-1) == ds.choice).float()),
        "corn": (pmf_corn.max(-1).values,
                 (pmf_corn.argmax(-1) == ds.severity - 1).float()),
    }
    strata = {"all": torch.ones(len(ds), dtype=torch.bool)}
    for k in (0, 1, 2):
        strata[{0: "calm", 1: "warm", 2: "hot"}[k]] = ds.stratum == k
    for s_name, m_stratum in strata.items():
        row: dict[str, float] = {}
        for head in ("gate", "choice", "corn"):
            conf, corr = heads[head]
            m = m_stratum & (ds.gate == 1) if head == "corn" else m_stratum
            m_n = m.cpu().numpy()
            if m_n.any():
                row[head] = smooth_ece(conf[m].cpu().numpy(), corr[m].cpu().numpy(),
                                       bandwidth=bw)
            else:
                row[head] = float("nan")
        out[s_name] = row
    return out


def train_student(
    net: CAMHDNet,
    replay: dict[str, ReplayDataset],
    targets: dict[str, torch.Tensor],
    tsc: TrainStudentConfig,
    run_cfg: RunConfig,
    stream_cfg: StreamConfig,
    teacher_temps: dict[str, float],
) -> StudentArtifact:
    """Train student-heads (+trunk) on the replay with the chosen loss.

`net` is the SAME model as the teacher (frozen broadheads); he
caller should have applied fresh set_requires_grad — the trainer will do it
strength here.
    """
    torch.manual_seed(run_cfg.train_seed)
    device = torch.device(tsc.device)
    net = net.to(device)
    ds = replay["train"].to(device)
    val = replay["val"].to(device)
    test = replay["test"].to(device)
    tg = {k: v.to(device) for k, v in targets.items()}

    net.set_requires_grad({"s", "trunk"})
    lv = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in _HN}).to(device)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in net.named_parameters() if p.requires_grad]},
        {"params": list(lv.values()), "weight_decay": 0.0},
    ], lr=tsc.lr, weight_decay=tsc.weight_decay)
    n = len(ds)
    steps = max(1, (n + tsc.batch_size - 1) // tsc.batch_size)
    total_steps = tsc.epochs * steps
    warm = max(1, total_steps // 10)

    def sched_fn(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    idx_perm = np.arange(n)
    tau = tsc.tau

    for epoch in range(tsc.epochs):
        np.random.default_rng(run_cfg.train_seed + 100 + epoch).shuffle(idx_perm)
        net.train()
        ep = {"loss": [], "gate_acc": []}
        for step in range(steps):
            idx = idx_perm[step * tsc.batch_size:(step + 1) * tsc.batch_size]
            idx_t = torch.from_numpy(idx)
            x, s = ds.features[idx_t], ds.s[idx_t]
            out = net(x, s, mode="student")["s"]
            att = ds.gate[idx_t] == 1

            # ground-truth losses (per head, corn masked to attacks)
            l_gate = F.cross_entropy(out["gate"], ds.gate[idx_t])
            l_choice = F.cross_entropy(out["choice"], ds.choice[idx_t])
            if att.any():
                rank = ds.severity[idx_t][att] - 1
                l_corn = corn_loss(out["corn"][att], rank)
            else:
                l_corn = out["corn"].sum() * 0.0
            base = sum(kendall_loss(lv[k], l) for k, l in
                       (("gate", l_gate), ("choice", l_choice), ("corn", l_corn)))

            if tsc.use_kd and tsc.alpha_mode != "none":
                z_s = {
                    "gate": out["gate"] / tau,
                    "choice": out["choice"] / tau,
                    "corn": out["corn"] / tau,   # logs for soft ordinal
                }
                # α(s): loading channel = q scaled (0..1), both sides at τ.
                # Phase B curriculum (plan §3.3): TIME 0 uses blend
                # global (α const) and from there the tail strata enter.
                if tsc.curriculum and epoch == 0:
                    a_s = torch.full_like(s[:, 0], tsc.alpha_base)
                elif tsc.alpha_mode == "const":
                    a_s = torch.full_like(s[:, 0], tsc.alpha_base)
                else:
                    s_frac = s[:, 0].clamp(0.0, 1.0)
                    a_s = tsc.alpha_base + (tsc.alpha_max - tsc.alpha_base) * s_frac
                kd_gate = -(tg["gate"][idx_t] * F.log_softmax(z_s["gate"], -1)).sum(-1)
                kd_choice = -(tg["choice"][idx_t] * F.log_softmax(z_s["choice"], -1)).sum(-1)
                # CORN: teacher targets are only valid in attacks
                # (his own training loss was masked); the KD of
                # corn if masks are the same — about benign they would be noise (K2)
                if att.any():
                    kd_corn = corn_soft_loss_from_pmf(z_s["corn"][att],
                                                      tg["corn"][idx_t][att])
                else:
                    kd_corn = out["corn"].sum() * 0.0
                loss_kd = ((a_s * kd_gate).mean()
                           + (a_s * kd_choice).mean())
                if att.any():
                    loss_kd = loss_kd + (a_s[att] * kd_corn).mean()
                base = base + loss_kd
                if tsc.lam_conf > 0:
                    conf_s = F.softmax(z_s["gate"], -1).max(-1).values
                    conf_t = tg["gate"][idx_t].max(-1).values
                    base = base + tsc.lam_conf * F.mse_loss(conf_s, conf_t.detach())
            elif not tsc.use_kd:
                pass  # B2: supervised only

            opt.zero_grad(set_to_none=True)
            base.backward()
            allp = [p for n2, p in net.named_parameters() if p.requires_grad] + list(lv.values())
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            sched.step()
            ep["loss"].append(float(base.item()))
            ep["gate_acc"].append(float((out["gate"].argmax(-1) == ds.gate[idx_t]).float().mean()))
        if tsc.log:
            print(f"[student:{tsc.variant_name()}] ep {epoch}: "
                  f"loss {np.mean(ep['loss']):.4f} gate_acc {np.mean(ep['gate_acc']):.4f}")

    # ---- student calibration (val) + metrics (test)
    with torch.no_grad():
        zval = net(val.features, val.s, mode="student")["s"]
    from .calibration import fit_temperatures
    val_att = val.gate == 1
    temps = fit_temperatures(
        {"gate": zval["gate"], "choice": zval["choice"], "corn": zval["corn"][val_att]},
        {"gate": val.gate, "choice": val.choice, "corn": val.severity[val_att] - 1})
    se = smooth_ece_strata(net, test, temps, tsc.smooth_bw, mode="student")
    with torch.no_grad():
        zt = net(test.features, test.s, mode="student")["s"]["gate"] / temps["gate"]
    from sklearn.metrics import roc_auc_score, average_precision_score
    pg = torch.softmax(zt, -1)[:, 1].numpy()
    metrics = {
        "smoothECE": se,
        "auroc_gate": float(roc_auc_score(test.gate.numpy(), pg)),
        "auprc_gate": float(average_precision_score(test.gate.numpy(), pg)),
        "variant": tsc.variant_name(),
    }
    if tsc.log:
        print(f"[student] temps: {json.dumps(temps)}")
        print(f"[student] smoothECE: {json.dumps(se, default=float)}")
    net.eval()
    return StudentArtifact(net=net.cpu(), arch=net.cfg, temperatures=dict(temps),
                           kendall_lv={k: float(v.item()) for k, v in lv.items()},
                           config=tsc, metrics=metrics, teacher_temps=teacher_temps)


def corn_soft_loss_from_pmf(student_corn_logits: torch.Tensor,
                            teacher_pmf_tau: torch.Tensor) -> torch.Tensor:
    """Distillation ordinal CE: the student logits are ALREADY tempered to
τ; the pmf of the target comes from the tempered teacher."""
    from .losses import corn_soft_loss
    return corn_soft_loss(student_corn_logits, teacher_pmf_tau)
