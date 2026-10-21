"""Configuraciones explícitas del proyecto (semillas siempre presentes).

Todo entrenamiento/corrida de simulación debe partir de un dataclass de este
módulo — sin globales mutables. Los valores por defecto son placeholders que
el pre-registro (CONTINUE.md §6) fijará antes del pilot del plan v3.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SmoothEceConfig:
    """Ancho de banda κ del kernel smoothECE.

    Nota G1 del plan v3: κ se elige por bias-variance del estimador con el
    n típico de ventana; es una cantidad PRE-REGISTRADA, no un tuneo mudo.
    """

    bandwidth: float = 0.10
    radius_scale: float = 5.0  # ventana = bandwidth * radius_scale
    include_self: bool = True


@dataclass(frozen=True)
class StreamConfig:
    """Generador de stream acoplado (parámetro γ, plan v3 §3.5).

    γ=0  → caso nulo de v2 (llegada independiente del contenido).
    γ=1  → acoplamiento total (los bursts inyectan ataque).
    La mezcla de clases α(t) sigue α(t) = alpha0 + gamma * (1 - alpha0) * burst(t).
    """

    d: int = 24                       # dimensión de features de flow
    n_choice_classes: int = 3         # subtipos de ataque (Choice multiclase)
    n_severity_levels: int = 3        # niveles ordinales CORN (1..3 sobre ataque)
    alpha0: float = 0.05              # cuota de ataque en régimen estable
    gamma: float = 0.5                # acoplamiento carga↔contenido
    period: float = 8.0               # periodo del régimen (s)

    # Punto operativo pre-registrado (F2): C_t medido en CPU con batch=16 y
    # service_scale=2.5 → C_t ≈ 12.7k ev/s: ρ_t(burst)=1.10 > 1 ≥ ρ_s=0.94.
    burst_duty: float = 0.25          # 2 s de burst por periodo de 8 s
    rate_idle: float = 2000.0         # llegadas/s en régimen estable
    rate_burst: float = 14000.0       # llegadas/s en burst
    attack_radius: float = 3.0        # ‖μ_ataque‖ — margen de Bayes Φ(−Δ/2)≈6.7%
    feature_scale: float = 1.0        # σ del ruido gaussiano
    severity_probs: tuple[float, float, float] = (0.55, 0.30, 0.15)
    seed: int = 0


@dataclass(frozen=True)
class RouterConfig:
    """Schmitt trigger por cabeza (plan v3 §3.4).

    El deadband (q_on − q_off) se deriva del κ_switch MEDIDO en el objetivo
    y del dinamismo de la cola — no de tuneo a ojo; el microbenchmark (scripts/)
    alimenta q_off antes de la campaña.
    """

    q_on: float = 0.80
    q_off: float = 0.50
    min_dwell: float = 2.0            # s mínimos entre conmutación (anti-flapping residual)
    latency_ema_alpha: float = 0.2    # suavizado del ℓ_p95 rodante
    latency_deadband: float = 0.15    # fracción de cambio mínimo para re-acción
    latency_refractory: float = 5.0   # s de congelación del canal tras conmutar
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    """Identidad de corrida: semillas separadas por subsistema (v3 §6)."""

    train_seed: int = 0
    stream_seed: int = 1
    bench_seed: int = 2
    tags: tuple[str, ...] = field(default_factory=tuple)
