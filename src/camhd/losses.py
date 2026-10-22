"""CA-MHD losses: ordinal CORN, homoscedastic multi-task (Kendall) and LCD.

Specificity of plan v3 (§3.2-3.3):
- CORN (Cao et al. 2021): chained conditionals q_1 = σ(z_1),
q_k = σ(z_k)/σ(z_{k-1}); pmf recovered for accumulated differences
(mathematically identical to the cumulative-product of conditionals).
- Kendall: homoscedastic weights 1/(2σ_q²) with regression of variance.
- LCD: L = Σ_q (1/2σ_q²)·L_q + ln(1+σ_q²) + α(s)·KL_τ(teacher″student) +
λ_conf·L_conf-cal, with KL at teacher→student address (distillation CE)
and τ IDENTICAL for all methods (pre-registration).

All the logit space used here is the SCALING (deployed): the
temperature is applied BEFORE rankings/softmax in calibrated projects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "corn_cond_probs",
    "corn_pmf",
    "corn_expected",
    "corn_loss",
    "corn_soft_loss",
    "kendall_loss",
    "LcdLoss",
    "kendall_total",
]

_EPS = 1e-7


# --------------------------------------------------------------------------
# Ordinal CORN (severity)
# --------------------------------------------------------------------------

def corn_cond_probs(corn_logits: torch.Tensor) -> torch.Tensor:
    """Conditional probabilities CORN q_k = P(y ≥ k | y ≥ k−1).

corn_logits: [B, C−1] (cumulative σ(z_k) = P(y ≥ k)).
Returns [B, C−1]: q_1 = σ(z_1); q_k = σ(z_k)/σ(z_{k−1}).
No clamp: conditionals ≥1 only occur with non-monotonic z's — the
clamping lives in corn_loss/corn_pmf, here we return the exact algebra
(cumprod(q) == σ(z) always).
    """
    s = torch.sigmoid(corn_logits)
    if s.size(-1) == 1:
        return s
    conds = torch.empty_like(s)
    conds[..., 0] = s[..., 0]
    conds[..., 1:] = s[..., 1:] / s[..., :-1].clamp_min(_EPS)
    return conds


def corn_pmf(corn_logits: torch.Tensor) -> torch.Tensor:
    """PMF of the ordinal law for accumulated differences σ: pmf_j = S_j − S_{j+1}.

Shape [B, C], sum 1 (clamping EPS for numerical stability with z
non-monotonic). Identical to cumprod of the CORN conditionals.
    """
    s = torch.sigmoid(corn_logits)
    upper = F.pad(s, (1, 0), value=1.0)   # S_j = P(y ≥ j): [1, s1, s2]
    lower = F.pad(s, (0, 1), value=0.0)   # [s1, s2, 0]
    pmf = (upper - lower).clamp_min(_EPS)
    return pmf / pmf.sum(dim=-1, keepdim=True)


def corn_expected(corn_logits: torch.Tensor) -> torch.Tensor:
    """E[Y] with levels 0..C−1 (expected score, golden section of the engine)."""
    pmf = corn_pmf(corn_logits)
    levels = torch.arange(pmf.size(-1), device=pmf.device, dtype=pmf.dtype)
    return pmf @ levels


def corn_loss(corn_logits: torch.Tensor, y_rank: torch.Tensor) -> torch.Tensor:
    """NLL CORN (Cao et al. 2021): ECB on eligible conditional units.

y_rank: levels 0..C−1 (0 = lowest level). Unit k (1..C−1) predicts
P(y ≥ k | y ≥ k−1); is eligible for samples with y ≥ k−1 and its target is
1[y ≥ k]. Conditional correction (division by previous cumulative)
avoids matching conditionals below eligibility.
    """
    if corn_logits.dim() != 2:
        raise ValueError(f"corn_logits debe ser [B, C−1], got {corn_logits.shape}")
    n_units = corn_logits.size(-1)
    c_classes = n_units + 1
    y = y_rank.long()
    if (y < 0).any() or (y >= c_classes).any():
        raise ValueError(f"y_rank fuera de [0, {c_classes - 1}]")
    s = torch.sigmoid(corn_logits)
    # cumulative targets per unit: t_k = 1[y ≥ k]
    ks = torch.arange(1, n_units + 1, device=y.device)
    targets = (y.unsqueeze(-1) >= ks.unsqueeze(0)).float()          # [B, C−1]
    eligible = (y.unsqueeze(-1) >= (ks - 1).unsqueeze(0)).float()   # [B, C−1]
    # conditional probabilities per unit
    q = torch.empty_like(s)
    q[..., 0] = s[..., 0]
    if n_units > 1:
        q[..., 1:] = s[..., 1:] / s[..., :-1].clamp_min(_EPS)
    q = q.clamp(_EPS, 1.0 - _EPS)
    log_like = (targets * torch.log(q) + (1.0 - targets) * torch.log1p(-q))
    return -(log_like * eligible).sum(-1).mean()


def corn_soft_loss(corn_logits: torch.Tensor, teacher_pmf: torch.Tensor) -> torch.Tensor:
    """Ordinal distillation with soft targets: −Σ_j pmf_T^(τ) · log pmf_S(level_j).

The teacher delivers his already tempered pmf to τ (the caller builds it).
    """
    pmf_s = corn_pmf(corn_logits)
    log_pmf_s = torch.log(pmf_s.clamp_min(_EPS))
    return -(teacher_pmf * log_pmf_s).sum(-1).mean()


# --------------------------------------------------------------------------
# Homoscedastic multi-task (Kendall et al. 2018)
# --------------------------------------------------------------------------

@dataclass
class KendallState:
    """Learnable state of variances per head (log σ²)."""

    log_vars: dict[str, float] = field(default_factory=dict)


def kendall_loss(log_var_q: torch.Tensor, loss_q: torch.Tensor) -> torch.Tensor:
    """(1/(2σ²))·L_q + ln σ → in lv = log σ²: e^{−lv}/2 · L_q + lv/2."""
    return 0.5 * torch.exp(-log_var_q) * loss_q + 0.5 * log_var_q


def kendall_total(log_vars: dict[str, torch.Tensor], losses: dict[str, torch.Tensor]) -> torch.Tensor:
    """Σ_q kendall: only the active heads (mask in the caller)."""
    total = 0.0
    for name, lv in log_vars.items():
        total = total + kendall_loss(lv, losses[name])
    return total


# --------------------------------------------------------------------------
# LCD — batch conditioned distillation
# --------------------------------------------------------------------------

class LcdLoss(nn.Module):
    """LCD Loss (Plan v3 §3.3 Phase B).

L = Σ_q (1/2σ_q²)·L_q + α(s)·KL_τ(T″S) + λ_conf·L_conf-cal

α(s): weight per sample derived from the state of charge (the shape is
ablatable; default = α_base + (α_max−α_base)·s_frac).
The teacher's logits enter warm to τ (the caller applies /τ).
The address is teacher→student (distillation CE: cross-entropy of S
against T's soft-target).
    """

    def __init__(self, tau: float = 2.0, lam_conf: float = 0.1, alpha_base: float = 0.5,
                 alpha_max: float = 2.0) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau debe ser > 0, got {tau}")
        if alpha_max < alpha_base:
            raise ValueError("alpha_max < alpha_base")
        self.tau = tau
        self.lam_conf = lam_conf
        self.alpha_base = alpha_base
        self.alpha_max = alpha_max

    def alpha(self, s_frac: torch.Tensor) -> torch.Tensor:
        """Default form of α(s): linear in the charge fraction s_frac ∈ [0,1]."""
        return self.alpha_base + (self.alpha_max - self.alpha_base) * s_frac

    @staticmethod
    def soft_ce(soft_target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Distillation EC with already tempered targets: −Σ p̂ log softmax(z).

The logits are not re-tempered here (the caller decides the scale).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_target * log_probs).sum(-1)

    def forward(
        self,
        s_frac: torch.Tensor,
        teacher_gate_target: torch.Tensor,
        teacher_choice_target: torch.Tensor,
        teacher_pmf: torch.Tensor,
        student_out: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Returns dict with the pieces per sample + those added.

s_frac: [B] in [0,1]; teacher targets [B,2] / [B,K+1] / [B,C] tempered;
student_out: dict gate/choice/corn logits of the student.
        """
        a_s = self.alpha(s_frac)

        kd_gate = self.soft_ce(teacher_gate_target, student_out["gate"])
        kd_choice = self.soft_ce(teacher_choice_target, student_out["choice"])
        kd_sev = corn_soft_loss(student_out["corn"], teacher_pmf)
        kd_total = kd_gate + kd_choice + kd_sev
        kd_weighted = a_s * kd_total

        # confidence calibration term: top-1 conf MSE among student
        # and teacher (scale unfolded; both already tempered)
        conf_s = torch.softmax(student_out["gate"], dim=-1).max(-1).values
        conf_t = teacher_gate_target.max(-1).values
        conf_cal = (conf_s - conf_t.detach()).pow(2)

        return {
            "kd_gate": kd_gate.mean(),
            "kd_choice": kd_choice.mean(),
            "kd_sev": kd_sev.mean(),
            "kd_weighted": kd_weighted.mean(),
            "conf_cal": conf_cal.mean(),
            "alpha_mean": a_s.mean(),
        }
