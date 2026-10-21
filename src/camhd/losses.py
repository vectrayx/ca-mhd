"""Pérdidas de CA-MHD: CORN ordinal, multi-task homoscedástico (Kendall) y LCD.

Especificidad del plan v3 (§3.2-3.3):
- CORN (Cao et al. 2021): condicionales encadenadas q_1 = σ(z_1),
  q_k = σ(z_k)/σ(z_{k-1}); pmf recuperada por diferencias de acumuladas
  (matemáticamente idéntica al cumulative-product de las condicionales).
- Kendall: pesos homoscedásticos 1/(2σ_q²) con regresión de varianza.
- LCD: L = Σ_q (1/2σ_q²)·L_q + ln(1+σ_q²) + α(s)·KL_τ(teacher‖student) +
  λ_conf·L_conf-cal, con KL en dirección teacher→student (CE de destilación)
  y τ IDÉNTICO para todos los métodos (pre-registro).

Todo el espacio de logits usado aquí es el ESCALADO (desplegado): la
temperatura se aplica ANTES de rankings/softmax en proyectos calibrados.
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
# CORN ordinal (severidad)
# --------------------------------------------------------------------------

def corn_cond_probs(corn_logits: torch.Tensor) -> torch.Tensor:
    """Probabilidades condicionales CORN q_k = P(y ≥ k | y ≥ k−1).

    corn_logits: [B, C−1] (acumuladas σ(z_k) = P(y ≥ k)).
    Retorna [B, C−1]: q_1 = σ(z_1); q_k = σ(z_k)/σ(z_{k−1}).
    Sin clamp: las condicionales ≥1 solo ocurren con z no-monótonas — el
    clamping vive en corn_loss/corn_pmf, aquí devolvemos el álgebra exacta
    (cumprod(q) == σ(z) siempre).
    """
    s = torch.sigmoid(corn_logits)
    if s.size(-1) == 1:
        return s
    conds = torch.empty_like(s)
    conds[..., 0] = s[..., 0]
    conds[..., 1:] = s[..., 1:] / s[..., :-1].clamp_min(_EPS)
    return conds


def corn_pmf(corn_logits: torch.Tensor) -> torch.Tensor:
    """PMF de la ley ordinal por diferencias de acumuladas σ: pmf_j = S_j − S_{j+1}.

    Shape [B, C], suma 1 (clamping eps para estabilidad numérica con z
    no-monótonas). Identical a cumprod de las condicionales CORN.
    """
    s = torch.sigmoid(corn_logits)
    upper = F.pad(s, (1, 0), value=1.0)   # S_j = P(y ≥ j): [1, s1, s2]
    lower = F.pad(s, (0, 1), value=0.0)   # [s1, s2, 0]
    pmf = (upper - lower).clamp_min(_EPS)
    return pmf / pmf.sum(dim=-1, keepdim=True)


def corn_expected(corn_logits: torch.Tensor) -> torch.Tensor:
    """E[Y] con niveles 0..C−1 (score esperado, sección dorada del engine)."""
    pmf = corn_pmf(corn_logits)
    levels = torch.arange(pmf.size(-1), device=pmf.device, dtype=pmf.dtype)
    return pmf @ levels


def corn_loss(corn_logits: torch.Tensor, y_rank: torch.Tensor) -> torch.Tensor:
    """NLL CORN (Cao et al. 2021): BCE sobre unidades condicionales elegibles.

    y_rank: niveles 0..C−1 (0 = nivel más bajo). Unidad k (1..C−1) predice
    P(y ≥ k | y ≥ k−1); es elegible para muestras con y ≥ k−1 y su target es
    1[y ≥ k]. La corrección condicional (división por la acumulada previa)
    evita acoplar las condicionales por debajo de la elegibilidad.
    """
    if corn_logits.dim() != 2:
        raise ValueError(f"corn_logits debe ser [B, C−1], got {corn_logits.shape}")
    n_units = corn_logits.size(-1)
    c_classes = n_units + 1
    y = y_rank.long()
    if (y < 0).any() or (y >= c_classes).any():
        raise ValueError(f"y_rank fuera de [0, {c_classes - 1}]")
    s = torch.sigmoid(corn_logits)
    # targets acumuladas por unidad: t_k = 1[y ≥ k]
    ks = torch.arange(1, n_units + 1, device=y.device)
    targets = (y.unsqueeze(-1) >= ks.unsqueeze(0)).float()          # [B, C−1]
    eligible = (y.unsqueeze(-1) >= (ks - 1).unsqueeze(0)).float()   # [B, C−1]
    # probabilidades condicionales por unidad
    q = torch.empty_like(s)
    q[..., 0] = s[..., 0]
    if n_units > 1:
        q[..., 1:] = s[..., 1:] / s[..., :-1].clamp_min(_EPS)
    q = q.clamp(_EPS, 1.0 - _EPS)
    log_like = (targets * torch.log(q) + (1.0 - targets) * torch.log1p(-q))
    return -(log_like * eligible).sum(-1).mean()


def corn_soft_loss(corn_logits: torch.Tensor, teacher_pmf: torch.Tensor) -> torch.Tensor:
    """Destilación ordinal con soft targets: −Σ_j pmf_T^(τ) · log pmf_S(level_j).

    El teacher entrega su pmf ya templada a τ (el caller la construye).
    """
    pmf_s = corn_pmf(corn_logits)
    log_pmf_s = torch.log(pmf_s.clamp_min(_EPS))
    return -(teacher_pmf * log_pmf_s).sum(-1).mean()


# --------------------------------------------------------------------------
# Multi-task homoscedástico (Kendall et al. 2018)
# --------------------------------------------------------------------------

@dataclass
class KendallState:
    """Estado aprendible de varianzas por cabeza (log σ²)."""

    log_vars: dict[str, float] = field(default_factory=dict)


def kendall_loss(log_var_q: torch.Tensor, loss_q: torch.Tensor) -> torch.Tensor:
    """(1/(2σ²))·L_q + ln σ  →  en lv = log σ²:  e^{−lv}/2 · L_q + lv/2."""
    return 0.5 * torch.exp(-log_var_q) * loss_q + 0.5 * log_var_q


def kendall_total(log_vars: dict[str, torch.Tensor], losses: dict[str, torch.Tensor]) -> torch.Tensor:
    """Σ_q kendall: solo las cabezas activas (mask en el caller)."""
    total = 0.0
    for name, lv in log_vars.items():
        total = total + kendall_loss(lv, losses[name])
    return total


# --------------------------------------------------------------------------
# LCD — destilación condicionada a carga
# --------------------------------------------------------------------------

class LcdLoss(nn.Module):
    """Pérdida LCD (plan v3 §3.3 Fase B).

    L = Σ_q (1/2σ_q²)·L_q + α(s)·KL_τ(T‖S) + λ_conf·L_conf-cal

    α(s): peso por muestra derivado del estado de carga (la forma es
    ablatable; default = α_base + (α_max−α_base)·s_frac).
    Los logits del teacher entran templados a τ (el caller aplica /τ).
    La dirección es teacher→student (CE de destilación: cross-entropy de S
    contra el soft-target de T).
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
        """Forma default de α(s): lineal en la fracción de carga s_frac ∈ [0,1]."""
        return self.alpha_base + (self.alpha_max - self.alpha_base) * s_frac

    @staticmethod
    def soft_ce(soft_target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """CE de destilación con targets ya templados: −Σ p̂ log softmax(z).

        Los logits NO se re-templar aquí (el caller decide la escala).
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
        """Retorna dict con las piezas por muestra + las agregadas.

        s_frac: [B] en [0,1]; teacher targets [B,2] / [B,K+1] / [B,C] templados;
        student_out: dict gate/choice/corn logits del student.
        """
        a_s = self.alpha(s_frac)

        kd_gate = self.soft_ce(teacher_gate_target, student_out["gate"])
        kd_choice = self.soft_ce(teacher_choice_target, student_out["choice"])
        kd_sev = corn_soft_loss(student_out["corn"], teacher_pmf)
        kd_total = kd_gate + kd_choice + kd_sev
        kd_weighted = a_s * kd_total

        # término de calibración de confianza: MSE de top-1 conf entre student
        # y teacher (escala desplegada; ambos ya templados)
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
