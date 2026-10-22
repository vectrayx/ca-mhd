"""Explicit project configurations (seeds always present).

All training/simulation runs must start from a dataclass in this module — no mutable globals. Default values are placeholders that pre-registration (CONTINUE.md §6) will set before the v3 plan pilot.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SmoothEceConfig:
    """Kernel smoothECE bandwidth κ.

    Note G1 of v3 plan: κ is chosen by bias-variance of the estimator with the
    typical window n; it is a PRE-REGISTERED quantity, not a silent tuning.
    """

    bandwidth: float = 0.10
    radius_scale: float = 5.0  # window = bandwidth * radius_scale
    include_self: bool = True


@dataclass(frozen=True)
class StreamConfig:
    """Coupled stream generator (parameter γ, plan v3 §3.5).

    γ=0  → v2 null case (arrival independent of content).
    γ=1  → total coupling (bursts inject attack).
    The class mixture α(t) follows α(t) = alpha0 + gamma * (1 - alpha0) * burst(t).
    """

    d: int = 24                       # dimension of flow features
    n_choice_classes: int = 3         # attack subtypes (multiclass Choice)
    n_severity_levels: int = 3        # CORN ordinal levels (1..3 over attack)
    alpha0: float = 0.05              # attack quota in steady regime
    gamma: float = 0.5                # load↔content coupling
    period: float = 8.0               # regime period (s)

    # Pre-registered operating point (F2): C_t measured on CPU with batch=16 and
    # service_scale=2.5 → C_t ≈ 12.7k ev/s: ρ_t(burst)=1.10 > 1 ≥ ρ_s=0.94.
    burst_duty: float = 0.25          # 2 s burst per 8 s period
    rate_idle: float = 2000.0         # arrivals/s in steady regime
    rate_burst: float = 14000.0       # arrivals/s in burst
    attack_radius: float = 3.0        # ‖μ_attack‖ — Bayes margin Φ(−Δ/2)≈6.7%
    feature_scale: float = 1.0        # σ of Gaussian noise
    severity_probs: tuple[float, float, float] = (0.55, 0.30, 0.15)
    seed: int = 0


@dataclass(frozen=True)
class RouterConfig:
    """Schmitt trigger per head (plan v3 §3.4).

    The deadband (q_on − q_off) is derived from the κ_switch MEASURED on the target
    and from queue dynamics — not from hand tuning; the microbenchmark (scripts/)
    feeds q_off before the campaign.
    """

    q_on: float = 0.80
    q_off: float = 0.50
    min_dwell: float = 2.0            # s minimum between switching (anti-flapping residual)
    latency_ema_alpha: float = 0.2    # smoothing of rolling ℓ_p95
    latency_deadband: float = 0.15    # fraction of minimum change for re-action
    latency_refractory: float = 5.0   # s of channel freezing after switching
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    """Run identity: seeds separated by subsystem (v3 §6)."""

    train_seed: int = 0
    stream_seed: int = 1
    bench_seed: int = 2
    tags: tuple[str, ...] = field(default_factory=tuple)
