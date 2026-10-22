"""CA-MHD architecture (v3 plan §3.1, M4 resolution).

- A shared pre-norm MLP trunk: ALWAYS runs and stays hot in the cache
(small κ_switch base).
- Teacher: wide heads (Gate BCE, Choice CE, Severity CORN).
- Student-mode: ONLY heads switch to narrow version (factor 1/8 per
defect). The order of downshift by head is by measured cost (microbench).
- Variant B (additional narrow trunk) is decided with the R5 measurement: if the
trunk dominates >70% of FLOPs, it is mandatory — the decision is AUTOMATIZABLE
here via `flops_share` and `use_student_trunk`.

Logit space: SCALING (unfolding) across theory and measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ArchConfig", "CAMHDNet", "HeadNames"]


@dataclass(frozen=True)
class ArchConfig:
    d: int = 24
    s_dim: int = 3                 # S_infra = (q_t, λ_t, ℓ_p95_t) ∈ R³ (input del trunk)
    w: int = 64                    # anchura del trunk (teacher)
    depth: int = 2                 # bloques del trunk
    student_frac: float = 0.125    # fraction of width of student heads
    head_mult: float = 4.0         # hidden width of teacher's heads = w·head_mult
    n_gate: int = 2                # benigno/ataque (BCE 2-logit softmax)
    n_choice: int = 4              # 0 benign + K-1 subtipos (Choice CE)
    n_sev: int = 3                 # ordinal levels CORN (units = n_sev−1)
    dropout: float = 0.0
    student_trunk_frac: float | None = None  # variante B: None = trunk compartido

    def trunk_input(self) -> int:
        return self.d + self.s_dim


class PreNormBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.fc(x))))


class CornHead(nn.Module):
    """Ordinal CORN: units = C−1 (cumulative σ(z_k) = P(y ≥ k))."""

    def __init__(self, w: int, units: int, bias_init: tuple[float, ...] = (1.0, 0.0, -1.0)) -> None:
        super().__init__()
        self.fc = nn.Linear(w, units, bias=True)
        if units <= len(bias_init):
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init[:units]))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def _mlp_head(w_in: int, w_hidden: int, out: int, dropout: float,
              corn_units: int | None = None) -> nn.Module:
    """Head: narrow (student) or direct bottleneck. CORN uses units
accumulated; for narrow mode, Linear(w_in→w_hidden) + CornHead."""
    if corn_units is not None:
        if w_hidden == w_in:
            return CornHead(w_in, corn_units)
        return nn.Sequential(PreNormBlock(w_in, w_hidden, dropout), CornHead(w_hidden, corn_units))
    return nn.Sequential(PreNormBlock(w_in, w_hidden, dropout), nn.Linear(w_hidden, out))


class CAMHDNet(nn.Module):
    """Single network with shared trunk + wide heads (teacher) + narrow heads (student).

forward(x, mode="both") → dict with keys "t"/"s" (teacher/student):
{"t": {gate, choice, corn}, "s": {gate, choice, corn}}
    """

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w_s = max(4, int(round(cfg.w * cfg.student_frac)))
        d_in = cfg.trunk_input()
        self.trunk = nn.ModuleList()
        self.trunk.append(PreNormBlock(d_in, cfg.w, cfg.dropout))
        for _ in range(cfg.depth - 1):
            self.trunk.append(PreNormBlock(cfg.w, cfg.w, cfg.dropout))

        # teacher heads (wide: hidden = w·head_mult — computation dominates
        # in serving; the DES gap teacher/student comes from here)
        h_t = int(round(cfg.w * cfg.head_mult))
        self.t_gate = _mlp_head(cfg.w, h_t, cfg.n_gate, cfg.dropout)
        self.t_choice = _mlp_head(cfg.w, h_t, cfg.n_choice, cfg.dropout)
        self.t_corn = _mlp_head(cfg.w, cfg.w, 0, cfg.dropout, corn_units=cfg.n_sev - 1)

        # student heads (narrow) — on the SAME trunk (variant A)
        self.s_gate = _mlp_head(cfg.w, w_s, cfg.n_gate, cfg.dropout)
        self.s_choice = _mlp_head(cfg.w, w_s, cfg.n_choice, cfg.dropout)
        self.s_corn = _mlp_head(cfg.w, w_s, 0, cfg.dropout, corn_units=cfg.n_sev - 1)

        # variant B (additional narrow trunk): optional
        self._student_trunk: nn.ModuleList | None = None
        if cfg.student_trunk_frac is not None:
            w_bt = max(4, int(round(cfg.w * cfg.student_trunk_frac)))
            bt: nn.ModuleList = nn.ModuleList([PreNormBlock(d_in, w_bt, cfg.dropout)])
            bt += [PreNormBlock(w_bt, w_bt, cfg.dropout) for _ in range(cfg.depth - 1)]
            self._student_trunk = bt
            # student heads rewired to narrow trunk
            self.s_gate = _mlp_head(w_bt, max(4, min(w_s, w_bt)), cfg.n_gate, cfg.dropout)
            self.s_choice = _mlp_head(w_bt, max(4, min(w_s, w_bt)), cfg.n_choice, cfg.dropout)
            self.s_corn = _mlp_head(w_bt, max(4, min(w_s, w_bt)), 0, cfg.dropout, corn_units=cfg.n_sev - 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None,
                mode: str = "both") -> dict[str, dict[str, torch.Tensor]]:
        """x: [B,d]; s: [B,s_dim] infra state (zeros if None)."""
        if s is None:
            s = torch.zeros(x.size(0), self.cfg.s_dim, device=x.device, dtype=x.dtype)
        h_in = torch.cat([x, s], dim=-1)
        h = h_in
        for blk in self.trunk:
            h = blk(h)
        out: dict[str, dict[str, torch.Tensor]] = {}
        if mode in ("teacher", "both"):
            out["t"] = {"gate": self.t_gate(h), "choice": self.t_choice(h), "corn": self.t_corn(h)}
        if mode in ("student", "both"):
            if self._student_trunk is not None:
                h_s = h_in
                for blk in self._student_trunk:
                    h_s = blk(h_s)
            else:
                h_s = h  # shared trunk ALWAYS HOT
            out["s"] = {"gate": self.s_gate(h_s), "choice": self.s_choice(h_s), "corn": self.s_corn(h_s)}
        return out

    def forward_heads(self, x: torch.Tensor, s: torch.Tensor | None = None,
                      narrow_heads: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        """SERVING path: shared trunk + each head in its mode.

narrow_heads: names of heads running in narrow version
(student). The Schmitt router per head controls this set.
        """
        if s is None:
            s = torch.zeros(x.size(0), self.cfg.s_dim, device=x.device, dtype=x.dtype)
        h_in = torch.cat([x, s], dim=-1)
        h = h_in
        for blk in self.trunk:
            h = blk(h)
        out: dict[str, torch.Tensor] = {}
        if self._student_trunk is not None and any(k in narrow_heads for k in ("gate", "choice", "corn")):
            h_s = h_in
            for blk in self._student_trunk:
                h_s = blk(h_s)
        else:
            h_s = h
        for name in ("gate", "choice", "corn"):
            if name in narrow_heads:
                out[name] = getattr(self, f"s_{name}")(h_s)
            else:
                out[name] = getattr(self, f"t_{name}")(h)
        return out

    # ------------------------------------------------------------------
    def teacher_params(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith("t_")]

    def student_params(self) -> list[nn.Parameter]:
        names = ("s_",) if self._student_trunk is None else ("s_", "_student_trunk")
        return [p for n, p in self.named_parameters() if n.startswith(names)]

    def trunk_params(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith("trunk")]

    def set_requires_grad(self, heads: set[str]) -> None:
        """Freeze/thaw by head for training consistent with
deployment automaton (G3 plan): {'t', 's', 'trunk'}.
        """
        for n, p in self.named_parameters():
            key = n.split(".", 1)[0].lstrip("_")
            if key.startswith("t_"):
                p.requires_grad = "t" in heads
            elif key.startswith("s_"):
                p.requires_grad = "s" in heads
            elif key.startswith("trunk"):
                p.requires_grad = "trunk" in heads
            elif key.startswith("student_trunk"):
                p.requires_grad = "s" in heads

    def param_count(self, which: str = "all") -> int:
        if which == "all":
            return sum(p.numel() for p in self.parameters())
        if which == "teacher_heads":
            return sum(p.numel() for p in self.teacher_params())
        if which == "student_heads":
            return sum(p.numel() for p in self.student_params())
        if which == "trunk":
            return sum(p.numel() for p in self.trunk_params())
        raise ValueError(which)

    def flops_forward(self, batch: int = 1) -> dict[str, int]:
        """Forward FLOPs (approximate MACs×2) count by mode — R5 input
(decision of variant B) and the table of FLOPs of the paper.
        """
        d_in = self.cfg.trunk_input()
        h_t = int(round(self.cfg.w * self.cfg.head_mult))

        def mlp_flops(d_i: int, d_hidden: int, out: int) -> int:
            return 2 * batch * (d_i * d_hidden + d_hidden * out)

        trunk_f = 2 * batch * (d_in * self.cfg.w + self.cfg.w * self.cfg.w)
        t_g = mlp_flops(self.cfg.w, h_t, self.cfg.n_gate)
        t_c = mlp_flops(self.cfg.w, h_t, self.cfg.n_choice)
        t_o = 2 * batch * (self.cfg.w * (self.cfg.n_sev - 1))
        teacher = trunk_f + t_g + t_c + t_o

        w_s = max(4, int(round(self.cfg.w * self.cfg.student_frac)))
        s_g = mlp_flops(self.cfg.w, w_s, self.cfg.n_gate)
        s_c = mlp_flops(self.cfg.w, w_s, self.cfg.n_choice)
        s_o = 2 * batch * (self.cfg.w * w_s) + 2 * batch * (w_s * (self.cfg.n_sev - 1))
        student = trunk_f + s_g + s_c + s_o
        return {"teacher": teacher, "student": student, "trunk": trunk_f}
