"""Phase A — Calibrated Teacher Training (Plan v3 §3.3).

- Kendall homoscedastic per head (gate/choice/corn) with mask of
activity in corn (attack samples only).
- Temperature scaling per head over VAL, SCALED space (deployed).
- Gate G1 criterion: smoothECE ≤ 1% per head and per load layer.
- Near-the-margin point (E4): the teacher is built with capacity
adjusted; the `attack_radius` of the stream controls the margin.
- Self-contained artifact: {net state, ArchConfig, StreamConfig, temps,
kendall lv, standardization, metrics}.
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

from .calibration import fit_temperatures
from .config import RunConfig, StreamConfig
from .dataset import ReplayDataset
from .losses import corn_loss, corn_pmf, kendall_loss
from .models import CAMHDNet, ArchConfig
from .smooth_ece import smooth_ece

__all__ = ["TrainTeacherConfig", "TeacherArtifact", "train_teacher"]


@dataclass
class TrainTeacherConfig:
    arch: ArchConfig
    epochs: int = 6
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-2
    smooth_bw: float = 0.05         # κ del kernel smoothECE (pre-registro G1)
    device: str = "cpu"
    log: bool = True


@dataclass
class TeacherArtifact:
    net: CAMHDNet
    arch: ArchConfig
    stream: StreamConfig
    run: RunConfig
    temperatures: dict[str, float]
    kendall_lv: dict[str, float]
    metrics: dict
    standardize: tuple[np.ndarray, np.ndarray] | None

    def save(self, path: str | Path) -> None:
        torch.save({
            "net_state": self.net.state_dict(),
            "arch": asdict(self.arch),
            "stream": asdict(self.stream),
            "run": asdict(self.run),
            "temperatures": self.temperatures,
            "kendall_lv": self.kendall_lv,
            "metrics": self.metrics,
            "standardize": (None if self.standardize is None else (
                self.standardize[0].tolist(), self.standardize[1].tolist())),
            "format": "camhd-teacher-v1",
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TeacherArtifact":
        blob = torch.load(str(path), weights_only=False)
        assert blob.get("format") == "camhd-teacher-v1", blob.get("format")
        arch = ArchConfig(**blob["arch"])
        net = CAMHDNet(arch)
        net.load_state_dict(blob["net_state"])
        from .config import StreamConfig as SC, RunConfig as RC
        std = None
        if blob["standardize"] is not None:
            std = (np.asarray(blob["standardize"][0]), np.asarray(blob["standardize"][1]))
        return cls(net=net, arch=arch, stream=SC(**blob["stream"]), run=RC(**blob["run"]),
                   temperatures=blob["temperatures"], kendall_lv=blob["kendall_lv"],
                   metrics=blob["metrics"], standardize=std)


def _batch(ds: ReplayDataset, idx: np.ndarray) -> dict[str, torch.Tensor]:
    return {"x": ds.features[idx], "s": ds.s[idx], "gate": ds.gate[idx], "choice": ds.choice[idx],
            "sev": ds.severity[idx]}


def _scheduler(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    return fn


def _teacher_losses(t_out: dict[str, torch.Tensor], batch) -> dict[str, torch.Tensor]:
    gate_ce = F.cross_entropy(t_out["gate"], batch["gate"])
    choice_ce = F.cross_entropy(t_out["choice"], batch["choice"])
    attack = batch["gate"] == 1
    if attack.any():
        rank = batch["sev"][attack] - 1
        corn_l = corn_loss(t_out["corn"][attack], rank)
    else:  # head idle in batch: zero tensor connected to graph
        corn_l = t_out["corn"].sum() * 0.0
    return {"gate": gate_ce, "choice": choice_ce, "corn": corn_l}


@torch.no_grad()
def _calibrated_pairs(net: CAMHDNet, ds: ReplayDataset, temps, head: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(conf, correctness, valid) to full length, unfolded space.
For 'corn' the non-attacks mark valid=False."""
    t = net(ds.features, ds.s, mode="teacher")["t"]
    valid = np.ones(len(ds), dtype=bool)
    if head == "gate":
        probs = torch.softmax(t["gate"] / temps["gate"], dim=-1)
        conf, pred = probs.max(-1).values, probs.argmax(-1)
        correct = (pred == ds.gate).float()
    elif head == "choice":
        probs = torch.softmax(t["choice"] / temps["choice"], dim=-1)
        conf, pred = probs.max(-1).values, probs.argmax(-1)
        correct = (pred == ds.choice).float()
    elif head == "corn":
        pmf = corn_pmf(t["corn"] / temps["corn"])
        conf_full, pred = pmf.max(-1).values, pmf.argmax(-1)
        y_rank = ds.severity - 1
        corr = (pred == y_rank).float()
        valid = (ds.gate == 1).numpy()      # ordinal head: attacks only
        return conf_full.cpu().numpy(), corr.cpu().numpy(), valid
    else:
        raise ValueError(head)
    return conf.cpu().numpy(), correct.cpu().numpy(), valid


def smooth_ece_by_head_stratum(net: CAMHDNet, ds: ReplayDataset, temps,
                               bw: float) -> dict[str, dict[str, float]]:
    """smoothECE per head × stratum (all + {calm, warm, hot} per quantiles
of q). CORN only attacks."""
    net.eval()
    out: dict[str, dict[str, float]] = {}
    strata = {"all": np.ones(len(ds), dtype=bool)}
    for k in (0, 1, 2):
        strata[{0: "calm", 1: "warm", 2: "hot"}[k]] = ds.stratum.numpy() == k
    for s_name, s_mask in strata.items():
        row: dict[str, float] = {}
        for head in ("gate", "choice", "corn"):
            conf, corr, valid = _calibrated_pairs(net, ds, temps, head)
            m = s_mask & valid
            row[head] = smooth_ece(conf[m], corr[m], bandwidth=bw) if bool(m.any()) else float("nan")
        out[s_name] = row
    net.train()
    return out


def train_teacher(
    replay: dict[str, ReplayDataset],
    ttc: TrainTeacherConfig,
    stream_cfg: StreamConfig,
    run_cfg: RunConfig | None = None,
) -> TeacherArtifact:
    run_cfg = run_cfg or RunConfig(train_seed=0, stream_seed=stream_cfg.seed)
    torch.manual_seed(run_cfg.train_seed)
    device = torch.device(ttc.device)
    net = CAMHDNet(ttc.arch).to(device).train()
    ds = replay["train"].to(device)
    val = replay["val"].to(device)
    test = replay["test"].to(device)
    n = len(ds)
    idx_perm = np.arange(n)

    lv_names = ["gate", "choice", "corn"]
    lv = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in lv_names}).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=ttc.lr, weight_decay=ttc.weight_decay)
    opt.add_param_group({"params": list(lv.values()), "weight_decay": 0.0})
    steps = max(1, (n + ttc.batch_size - 1) // ttc.batch_size)
    total_steps = ttc.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _scheduler(total_steps, max(1, total_steps // 20)))

    for epoch in range(ttc.epochs):
        np.random.default_rng(run_cfg.train_seed + epoch).shuffle(idx_perm)
        net.train()
        ep_losses, ep_gate_acc = [], []
        for step in range(steps):
            idx = idx_perm[step * ttc.batch_size:(step + 1) * ttc.batch_size]
            batch = _batch(ds, idx)
            t_out = net(batch["x"], batch["s"], mode="teacher")["t"]
            losses = _teacher_losses(t_out, batch)
            total = sum(kendall_loss(lv[k], losses[k]) for k in losses)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(lv.values()), 1.0)
            opt.step()
            sched.step()
            ep_losses.append(float(total.item()))
            ep_gate_acc.append(float((t_out["gate"].argmax(-1) == batch["gate"]).float().mean()))
        if ttc.log:
            print(f"[teacher] ep {epoch}: loss {np.mean(ep_losses):.4f} "
                  f"gate_acc {np.mean(ep_gate_acc):.4f}")

    # calibration (val) + metrics (test)
    # Amendment 2: T per head minimizing the smoothECE of val (the criterion
    # G1 of the pre-registration); the argmax is invariant at T (positive scale).
    with torch.no_grad():
        tv = net(val.features, val.s, mode="teacher")["t"]
    val_attack = val.gate == 1
    temps = fit_temperatures(
        {"gate": tv["gate"], "choice": tv["choice"], "corn": tv["corn"][val_attack]},
        {"gate": val.gate, "choice": val.choice, "corn": val.severity[val_attack] - 1},
        objective="smooth_ece")
    se = smooth_ece_by_head_stratum(net, test, temps, ttc.smooth_bw)
    with torch.no_grad():
        tg = net(test.features, test.s, mode="teacher")["t"]["gate"] / temps["gate"]
    from sklearn.metrics import average_precision_score, roc_auc_score
    p_gate = torch.softmax(tg, -1)[:, 1].numpy()
    metrics = {
        "smoothECE": se,
        "auroc_gate": float(roc_auc_score(test.gate.numpy(), p_gate)),
        "auprc_gate": float(average_precision_score(test.gate.numpy(), p_gate)),
        "final_train_loss": float(np.mean(ep_losses[-steps:])),
    }
    if ttc.log:
        print("[teacher] temps:", json.dumps(temps))
        print("[teacher] smoothECE:", json.dumps(se, default=float))
        print(f"[teacher] AUROC {metrics['auroc_gate']:.4f} AUPRC {metrics['auprc_gate']:.4f}")
    net.eval()
    return TeacherArtifact(net=net.cpu(), arch=ttc.arch, stream=stream_cfg, run=run_cfg,
                           temperatures=dict(temps),
                           kendall_lv={k: float(v.item()) for k, v in lv.items()},
                           metrics=metrics, standardize=replay["train"].standardize)
