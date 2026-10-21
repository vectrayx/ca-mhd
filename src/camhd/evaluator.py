"""Evaluador desplegable (F3/F4): métricas del CUL-Bench por estrato.

De un SimResult (DES con modelo y política) + tiles de q del glass box train,
produce:
- smoothECE desplegado por cabeza × estrato (calm/warm/hot) — el δ(s)
  contrarresta el observado por el contrato;
- accuracy/AUROC por estrato del gate;
- κ contable: nº de conmutaciones, cobertura student, tristeza Δ vs el
  teacher-only (estimador de captura de Prop 3),
- violacion SLA y percentiles de latencia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bench.sim import SimResult
from .smooth_ece import smooth_ece

__all__ = ["deployed_metrics", "StrataTiles"]


@dataclass(frozen=True)
class StrataTiles:
    q50: float = 1.0
    q90: float = 2.0

    def of(self, q: np.ndarray) -> np.ndarray:
        return np.digitize(q, bins=(self.q50, self.q90), right=True)


def deployed_metrics(
    res: SimResult,
    tiles: StrataTiles,
    labels_gate: np.ndarray,
    labels_choice: np.ndarray,
    labels_sev: np.ndarray,
    l_max_us: float = 1500.0,
    bw: float = 0.05,
) -> dict:
    """Métricas del DES desplegado. Los arrays de labels son del split evaluado."""
    n = res.n
    q = res.s_vec[:, 0] if n else np.zeros(0)
    stratum = tiles.of(q) if n else np.zeros(0, dtype=np.int64)
    p_gate = res.pred_gate if n else np.zeros(0)
    conf_gate = np.maximum(p_gate, 1.0 - p_gate)
    corr_gate = ((p_gate >= 0.5).astype(np.int64) == labels_gate[:n]).astype(float)
    corr_choice = (res.pred_choice[:n] == labels_choice[:n]).astype(float)
    corr_corn = np.zeros(n)
    m_corn = (labels_gate[:n] == 1)  # la cabeza ordinal solo evaluable en ataques
    corr_corn[m_corn] = (res.pred_level[:n][m_corn] == labels_sev[:n][m_corn] - 1)
    conf_corn = res.conf_corn[:n]

    out: dict = {"smoothECE": {}, "accuracy": {}, "gate": {}}
    for s in (0, 1, 2):
        m_s = stratum == s
        row = {}
        if m_s.any():
            row["gate"] = smooth_ece(conf_gate[m_s], corr_gate[m_s], bandwidth=bw)
            row["choice"] = smooth_ece(res.conf_choice[:n][m_s], corr_choice[m_s], bandwidth=bw)
            m_corn_s = m_s & m_corn
            row["corn"] = (smooth_ece(conf_corn[m_corn_s], corr_corn[m_corn_s], bandwidth=bw)
                           if m_corn_s.any() else float("nan"))
        out["smoothECE"][{0: "calm", 1: "warm", 2: "hot"}[s]] = row
        acc = {}
        if m_s.any():
            acc["gate"] = float(corr_gate[m_s].mean())
            acc["choice"] = float(corr_choice[m_s].mean())
        out["accuracy"][{0: "calm", 1: "warm", 2: "hot"}[s]] = acc
    out["gate"] = {
        "auroc": _auroc(p_gate, labels_gate[:n]),
        "acc": float(corr_gate.mean()) if n else float("nan"),
    }
    # κ contable y cobertura student
    out["serve"] = {
        "n_switches": int(res.n_switches),
        "student_coverage": float(res.narrow[:n].any(axis=1).mean()) if n else float("nan"),
        "lat_mean_us": res.summary["lat_mean_us"],
        "lat_p50_us": float(np.percentile(res.lat_us[:n], 50)) if n else float("nan"),
        "lat_p95_us": res.summary["lat_p95_us"],
        "lat_p99_us": res.summary["lat_p99_us"],
        "sla_viol_rate": res.summary["sla_viol_rate"],
        "q_p95": res.summary["q_p95"],
        "q_max": res.summary["q_max"],
    }
    return out


def _auroc(p: np.ndarray, y: np.ndarray) -> float:
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
