"""Temperature scaling por cabeza sobre el espacio desplegado (plan v3 §3.1).

- Gate/Choice: 1D T, CE sobre el val de calibración.
- CORN: T conjunta sobre las unidades (severidad).
Grid log-uniforme burdo + sección dorada (tol ~1e-3). El artefacto guarda
T por cabeza; TODA la teoría/medición usa los logits escalados.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["Temperatures", "fit_temperatures"]


class Temperatures(dict):
    """Mapping head → temperatura escalar (arde a [0.05, 20])."""

    def clamp(self) -> "Temperatures":
        return Temperatures({k: float(min(20.0, max(0.05, v))) for k, v in self.items()})


def _grid_golden(loss_fn, lo: float = 0.05, hi: float = 20.0, n_grid: int = 200,
                 tol: float = 1e-3) -> float:
    """Grid log-uniforme en T + refinamiento por sección dorada."""
    ts = torch.exp(torch.linspace(math.log(lo), math.log(hi), n_grid))
    with torch.no_grad():
        losses = [loss_fn(float(t)) for t in ts]
    best_i = int(torch.tensor(losses).argmin())
    lo_b = lo if best_i == 0 else float(ts[best_i - 1])
    hi_b = hi if best_i == n_grid - 1 else float(ts[best_i + 1])

    gr = (5 ** 0.5 - 1) / 2
    a, b = lo_b, hi_b
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = loss_fn(c), loss_fn(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = loss_fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = loss_fn(d)
    return (a + b) / 2


def fit_temperatures(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    objective: str = "nll",
    bandwidth: float = 0.05,
) -> Temperatures:
    """Ajusta T por cabeza en val.

    objective="nll": minimiza la NLL de calibración (original, Guo et al.).
    objective="smooth_ece": minimiza el smoothECE de val (el criterio G1 del
    pre-registro). El argmax es INVARIANTE a T (escala positiva), así que la
    re-optimización solo repara la CONFIANZA top-label — exactamente el
    objeto del contrato. CORN usa la loss ordinal o el smoothECE del pmf.
    """
    from .losses import corn_loss, corn_pmf
    from .smooth_ece import smooth_ece

    if objective not in ("nll", "smooth_ece"):
        raise ValueError(f"objective inválido: {objective}")

    def gate_loss(t: float) -> float:
        if objective == "nll":
            return float(F.cross_entropy(logits["gate"] / t, targets["gate"]).item())
        p = torch.softmax(logits["gate"] / t, -1)[:, 1]
        conf = torch.maximum(p, 1.0 - p)
        correct = ((p >= 0.5).float() == targets["gate"]).float()
        return smooth_ece(conf, correct, bandwidth=bandwidth)

    def choice_loss(t: float) -> float:
        if objective == "nll":
            return float(F.cross_entropy(logits["choice"] / t, targets["choice"]).item())
        probs = torch.softmax(logits["choice"] / t, -1)
        conf = probs.max(-1).values
        correct = (probs.argmax(-1) == targets["choice"]).float()
        return smooth_ece(conf, correct, bandwidth=bandwidth)

    def corn_t_loss(t: float) -> float:
        if objective == "nll":
            return float(corn_loss(logits["corn"] / t, targets["corn"]).item())
        pmf = corn_pmf(logits["corn"] / t)
        conf = pmf.max(-1).values
        correct = (pmf.argmax(-1) == targets["corn"]).float()
        return smooth_ece(conf, correct, bandwidth=bandwidth)

    ts = Temperatures()
    if "gate" in logits:
        ts["gate"] = _grid_golden(gate_loss)
    if "choice" in logits:
        ts["choice"] = _grid_golden(choice_loss)
    if "corn" in logits:
        ts["corn"] = _grid_golden(corn_t_loss)
    return ts.clamp()
