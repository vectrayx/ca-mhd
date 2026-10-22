"""↔ Content search coupled stream generator (plan v3 §3.5).

The part that corrects the root defect of v2: `s_infra` is not independent of
`x`. The arrival process (NHPP by chunks) modulates the class mix:

α(t) = alpha0 + gamma * (1 − alpha0) * burst(t)

with `burst(t)` square wave (period/duty). γ=0 reproduces the null case
(content independent arrival); γ=1 total coupling.

The features are all-Gaussian-relevant synthetics in Phase 0, with the
separation of effects controlled by `attack_radius` (protocol
near-the-margin: SMALL separation so that the margin mass is not zero).
The interface (`sample`/`generate`) is stable so that Phase F4 can
replace the synthetic sampler with real flows (CIC/UNSW) without touching the harness.
"""

from __future__ import annotations

import numpy as np

from .config import StreamConfig

__all__ = ["CoupledStream", "StreamSample"]


def _burst(t: np.ndarray, period: float, duty: float) -> np.ndarray:
    phase = t % period
    return (phase < duty * period).astype(bool) if duty > 0 else np.zeros(t.shape, dtype=bool)


class StreamSample:
    """Batch of stream events sorted by arrival (non-decreasing t)."""

    __slots__ = ("t", "features", "gate", "choice", "severity", "in_burst")

    def __init__(self, t: np.ndarray, features: np.ndarray, gate: np.ndarray, choice: np.ndarray,
                 severity: np.ndarray, in_burst: np.ndarray) -> None:
        self.t = t
        self.features = features
        self.gate = gate          # 1 = attack (binary head)
        self.choice = choice      # 0 = benign; 1..K attack subtypes
        self.severity = severity  # 0 = n/a for benign; 1..3 ordinal levels CORN
        self.in_burst = in_burst

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def attack_share(self) -> float:
        return float(self.gate.mean()) if len(self) else float("nan")


class CoupledStream:
    """Event source with NHPP arrivals by chunks and γ-coupled mix."""

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
        self._t = 0.0  # internal arrival clock
        # equispaced class centers in the plane (e1,e2): separation
        # radial·2·sin(π/K) — learnable, near-margin by construction
        ks = np.arange(cfg.n_choice_classes)
        theta = 2.0 * np.pi * ks / cfg.n_choice_classes
        centers = np.zeros((cfg.n_choice_classes, cfg.d))
        centers[:, 0] = np.cos(theta)
        centers[:, 1] = np.sin(theta)
        self._attack_centers = centers * cfg.attack_radius
        # ordinal severity signal: e3 axis + pre-recorded slope
        # (near-margin difficulty: Bayes-acc with β=2 over z~N(0.1) ≈ 0.72)
        self.sev_axis = 2
        self.sev_beta = 2.0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)
        self._t = 0.0

    def rate_at(self, t: float) -> float:
        """Instantaneous arrival rate λ(t)."""
        return self.cfg.rate_burst if _burst(np.array([t]), self.cfg.period, self.cfg.burst_duty)[0] else self.cfg.rate_idle

    def attack_share_at(self, t: float) -> float:
        """Mixture of classes α(t) — the heart of the coupling γ."""
        burst = bool(_burst(np.array([t]), self.cfg.period, self.cfg.burst_duty)[0])
        return self.cfg.alpha0 + self.cfg.gamma * (1.0 - self.cfg.alpha0) * (1.0 if burst else 0.0)

    def _draw_event(self, t: float) -> tuple[np.ndarray, int, int, int]:
        share = self.attack_share_at(t)
        is_attack = self._rng.random() < share
        if is_attack:
            k = int(self._rng.integers(0, self.cfg.n_choice_classes))
            x = self._attack_centers[k] + self._rng.standard_normal(self.cfg.d) * self.cfg.feature_scale
            choice = 1 + k
            # oRDINAL severity with real signal of unit variance: the
            # accumulated S_k(z) = P(y ≥ k) = σ(b·z − c_k) grow with z
            # (exact ordinal monotony via single threshold u ~ U(0,1)).
            # Note: logit_j + β·z is softmax-invariant (common shift) — ignored.
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
        """NHPP by chunks via thinning: proposals Exp(λ_max), accepted with
p = λ(t)/λ_max. Stop per COUNT (the temporary exposure of each regimen
is stochastic; for deterministic exposure use `generate_until`).
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
                # rejection due to thinness in steady state: p = rate_idle / rate_burst
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
        """Generates events up to a time horizon T (not inclusive).

The exposure of each regimen is DETERMINISTIC (integer multiples of
period + rest truncated): check arrival rates against duty
nominal is only valid with this API.
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
                break  # horizon censorship: the censoring process is still a correct NHPP
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
