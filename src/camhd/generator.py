"""Generador de stream acoplado búsqueda ↔ contenido (plan v3 §3.5).

La pieza que corrige el defecto raíz de v2: `s_infra` NO es independiente de
`x`. El proceso de llegada (NHPP por trozos) modula la mezcla de clases:

    α(t) = alpha0 + gamma * (1 − alpha0) * burst(t)

con `burst(t)` onda cuadrada (periodo/duty). γ=0 reproduce el caso nulo
(llegada independiente del contenido); γ=1 acoplamiento total.

Los features son sintéticos todo-relevantes-gaussianos en Fase 0, con la
separación de efectos controlada por `attack_radius` (protocolo
near-the-margin: separación CHICA para que la masa de margen no sea cero).
La interfaz (`sample`/`generate`) es estable para que Fase F4 pueda
sustituir el sampler sintético por flows reales (CIC/UNSW) sin tocar el harness.
"""

from __future__ import annotations

import numpy as np

from .config import StreamConfig

__all__ = ["CoupledStream", "StreamSample"]


def _burst(t: np.ndarray, period: float, duty: float) -> np.ndarray:
    phase = t % period
    return (phase < duty * period).astype(bool) if duty > 0 else np.zeros(t.shape, dtype=bool)


class StreamSample:
    """Batch de eventos de stream ordenados por llegada (t no decreciente)."""

    __slots__ = ("t", "features", "gate", "choice", "severity", "in_burst")

    def __init__(self, t: np.ndarray, features: np.ndarray, gate: np.ndarray, choice: np.ndarray,
                 severity: np.ndarray, in_burst: np.ndarray) -> None:
        self.t = t
        self.features = features
        self.gate = gate          # 1 = ataque (cabeza binaria)
        self.choice = choice      # 0 = benigno; 1..K subtipos de ataque
        self.severity = severity  # 0 = n/a para benigno; 1..3 niveles ordinales CORN
        self.in_burst = in_burst

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def attack_share(self) -> float:
        return float(self.gate.mean()) if len(self) else float("nan")


class CoupledStream:
    """Fuente de eventos con llegadas NHPP por trozos y mezcla γ-acoplada."""

    def __init__(self, cfg: StreamConfig) -> None:
        if not 0.0 <= cfg.gamma <= 1.0:
            raise ValueError(f"gamma debe estar en [0,1], got {cfg.gamma}")
        if cfg.rate_burst < cfg.rate_idle:
            raise ValueError("rate_burst < rate_idle: no habría hip operativa a estraddle-1")
        if len(cfg.severity_probs) != cfg.n_severity_levels:
            raise ValueError("severity_probs debe tener n_severity_levels entradas")
        total = sum(cfg.severity_probs)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"severity_probs debe sumar 1, got {total}")
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._t = 0.0  # reloj de llegada interno
        # centros de clase equiespaciados en el plano (e1,e2): separación
        # radial·2·sin(π/K) — aprendible, near-margin por construction
        ks = np.arange(cfg.n_choice_classes)
        theta = 2.0 * np.pi * ks / cfg.n_choice_classes
        centers = np.zeros((cfg.n_choice_classes, cfg.d))
        centers[:, 0] = np.cos(theta)
        centers[:, 1] = np.sin(theta)
        self._attack_centers = centers * cfg.attack_radius
        # señal ordinal de severidad: eje e3 + pendiente pre-registrada
        # (dificultad near-margin: la Bayes-acc con β=2 sobre z~N(0,1) ≈ 0.72)
        self.sev_axis = 2
        self.sev_beta = 2.0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)
        self._t = 0.0

    def rate_at(self, t: float) -> float:
        """Tasa instantánea de llegada λ(t)."""
        return self.cfg.rate_burst if _burst(np.array([t]), self.cfg.period, self.cfg.burst_duty)[0] else self.cfg.rate_idle

    def attack_share_at(self, t: float) -> float:
        """Mezcla de clases α(t) — el corazón del acoplamiento γ."""
        burst = bool(_burst(np.array([t]), self.cfg.period, self.cfg.burst_duty)[0])
        return self.cfg.alpha0 + self.cfg.gamma * (1.0 - self.cfg.alpha0) * (1.0 if burst else 0.0)

    def _draw_event(self, t: float) -> tuple[np.ndarray, int, int, int]:
        share = self.attack_share_at(t)
        is_attack = self._rng.random() < share
        if is_attack:
            k = int(self._rng.integers(0, self.cfg.n_choice_classes))
            x = self._attack_centers[k] + self._rng.standard_normal(self.cfg.d) * self.cfg.feature_scale
            choice = 1 + k
            # severidad ORDINAL con señal real de varianza unitaria: las
            # acumuladas S_k(z) = P(y ≥ k) = σ(b·z − c_k) crecen con z
            # (monotonía ordinal exacta vía umbral único u ~ U(0,1)).
            # Nota: logit_j + β·z es softmax-invariante (shift común) — ignorado.
            z_sev = float(x[self.sev_axis] - self._attack_centers[k][self.sev_axis])
            p = np.asarray(self.cfg.severity_probs, dtype=np.float64)
            s_cum = np.cumsum(p[::-1])[::-1][1:]   # P(≥2), P(≥3), ...
            cks = np.log(1.0 / s_cum[: self.cfg.n_severity_levels - 1] - 1.0)
            s_k = 1.0 / (1.0 + np.exp(cks[: self.cfg.n_severity_levels - 1] - self.sev_beta * z_sev))
            severity = 1 + int(np.sum(self._rng.random() <= s_k))
            gate = 1
        else:
            x = self._rng.standard_normal(self.cfg.d) * self.cfg.feature_scale
            choice, severity, gate = 0, 0, 0
        return x, gate, choice, severity

    def generate(self, n_events: int) -> StreamSample:
        """NHPP por trozos vía thinning: propuestas Exp(λ_max), aceptadas con
        p = λ(t)/λ_max. Para por CONTEO (la exposición temporal de cada régimen
        es estocástica; para exposición determinista usar `generate_until`).
        """
        if n_events <= 0:
            raise ValueError(f"n_events debe ser > 0, got {n_events}")
        cfg = self.cfg
        lam_max = cfg.rate_burst
        ts, feats, gates, choices, sevs, bursts = [], [], [], [], [], []
        got = 0
        while got < n_events:
            self._t += float(self._rng.exponential(1.0 / lam_max))
            t = self._t
            if not _burst(np.array([t]), cfg.period, cfg.burst_duty)[0]:
                # rechazo por delgadez en régimen estable: p = rate_idle / rate_burst
                if self._rng.random() > cfg.rate_idle / lam_max:
                    continue
            x, gate, choice, severity = self._draw_event(t)
            ts.append(t)
            feats.append(x)
            gates.append(gate)
            choices.append(choice)
            sevs.append(severity)
            bursts.append(bool(_burst(np.array([t]), cfg.period, cfg.burst_duty)[0]))
            got += 1
        return StreamSample(
            t=np.asarray(ts),
            features=np.asarray(feats),
            gate=np.asarray(gates, dtype=np.int64),
            choice=np.asarray(choices, dtype=np.int64),
            severity=np.asarray(sevs, dtype=np.int64),
            in_burst=np.asarray(bursts, dtype=bool),
        )

    def generate_until(self, t_end: float) -> StreamSample:
        """Genera eventos hasta un horizonte temporal T (no inclusive).

        La exposición de cada régimen es DETERMINISTA (múltiplos enteros del
        periodo + resto truncado): comprobar tasas de llegada contra duty
        nominal solo es válido con esta API.
        """
        if t_end <= 0:
            raise ValueError(f"t_end debe ser > 0, got {t_end}")
        cfg = self.cfg
        lam_max = cfg.rate_burst
        ts, feats, gates, choices, sevs, bursts = [], [], [], [], [], []
        while self._t < t_end:
            self._t += float(self._rng.exponential(1.0 / lam_max))
            t = self._t
            if t >= t_end:
                break  # censura del horizonte: el proceso censor es aún un NHPP correcto
            burst = bool(_burst(np.array([t]), cfg.period, cfg.burst_duty)[0])
            if not burst and self._rng.random() > cfg.rate_idle / lam_max:
                continue
            x, gate, choice, severity = self._draw_event(t)
            ts.append(t)
            feats.append(x)
            gates.append(gate)
            choices.append(choice)
            sevs.append(severity)
            bursts.append(burst)
        return StreamSample(
            t=np.asarray(ts),
            features=np.asarray(feats),
            gate=np.asarray(gates, dtype=np.int64),
            choice=np.asarray(choices, dtype=np.int64),
            severity=np.asarray(sevs, dtype=np.int64),
            in_burst=np.asarray(bursts, dtype=bool),
        )
