"""Router con histéresis (Schmitt trigger) + gobernanza del canal de latencia.

Diseño del plan v3 §3.4 (correcciones G4):
- Autómata por cabeza REUSABLE con zona muerta [q_off, q_on).
- `min_dwell`: tiempo mínimo en modo tras conmutar (anti-flapping residual
  que la zona muerta NO cubre: q oscilando por encima de q_on con picos no
  debe conmutar y des-comutar en milisegundos).
- `LatencyGuard`: ℓ_p95 rodante con EMA + deadband de cambio mínimo +
  periodo refractario tras cada conmutación — protege contra el loop
  oscilario: switch → spike de latencia (κ real) → ℓ_p95 dispara → switch
  de vuelta.
- El costo online es solo lo observable (violación SLA + conmutación), Prop 3
  v3: el registrado aquí es exactamente esa señal contable.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from enum import Enum

from .config import RouterConfig

__all__ = ["Mode", "SwitchRecord", "SchmittTrigger", "LatencyGuard"]


class Mode(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


@dataclass(frozen=True)
class SwitchRecord:
    t: float
    queue: float
    frm: Mode
    to: Mode
    reason: str  # "queue" | "manual"


class SchmittTrigger:
    """Autómata TEACHER→STUDENT con histéresis doble y dwell mínimo."""

    def __init__(self, cfg: RouterConfig, name: str = "head") -> None:
        if not 0.0 <= cfg.q_off < cfg.q_on <= 1.0:
            raise ValueError(f"requiere 0 ≤ q_off < q_on ≤ 1: got q_off={cfg.q_off}, q_on={cfg.q_on}")
        self.cfg = cfg
        self.name = name
        self._mode = Mode.TEACHER
        self._last_change_t: float | None = None
        self.records: list[SwitchRecord] = []

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def in_student(self) -> bool:
        return self._mode is Mode.STUDENT

    def _can_switch(self, t: float) -> bool:
        if self._last_change_t is None:
            return True
        return (t - self._last_change_t) >= self.cfg.min_dwell

    def _switch(self, t: float, q: float, reason: str) -> bool:
        frm = self._mode
        self._mode = Mode.STUDENT if frm is Mode.TEACHER else Mode.TEACHER
        self._last_change_t = t
        self.records.append(SwitchRecord(t=t, queue=q, frm=frm, to=self._mode, reason=reason))
        return True

    def update(self, q: float, t: float) -> bool:
        """Evalúa el trigger con la profundidad de cola actual. Retorna True
        ssi conmutó (el caller debe contabilizar el costo κ_switch).
        """
        if not np.isfinite(q):
            return False
        if self._mode is Mode.TEACHER and q >= self.cfg.q_on and self._can_switch(t):
            return self._switch(t, q, "queue")
        if self._mode is Mode.STUDENT and q <= self.cfg.q_off and self._can_switch(t):
            return self._switch(t, q, "queue")
        return False

    @property
    def switch_count(self) -> int:
        return len(self.records)

    def student_coverage(self, t_end: float | None = None) -> float:
        """Fracción temporal en modo student desde t=0."""
        if not self.records:
            return 0.0
        end = t_end if t_end is not None else self.records[-1].t
        if end <= 0:
            return 0.0
        total = 0.0
        # recorrer intervalos: student desde el t de switch-to-student hasta el
        # siguiente switch-to-teacher (o end)
        cur_start: float | None = None
        for r in self.records:
            if r.to == Mode.STUDENT:
                cur_start = r.t
            elif cur_start is not None:
                total += r.t - cur_start
                cur_start = None
        if cur_start is not None:
            total += end - cur_start
        return total / end


class LatencyGuard:
    """ digsCondición del canal de latencia: EMA de ℓ_p95 + deadband +
    periodo refractario tras conmutación. No toma decisiones — entrega
    `saturated_signal` que el调用 router puede usar para bloquear acciones
    motivadas por latencia mientras el pico por κ_switch se metaboliza.
    """

    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg
        self._ema: float | None = None
        self._last_switch_t: float = float("-inf")
        self.baseline: float | None = None  # ℓ de referencia (régimen estable)

    def on_switch(self, t: float) -> None:
        self._last_switch_t = t

    def feeding(self, lp95: float, t: float) -> float | None:
        """Actualiza la señal rodante; retorna Δ normalizado contra baseline
        (0 si aún no hay baseline). None si el canal está en refractario.
        """
        if not (np.isfinite(lp95) and np.isfinite(t)):
            raise ValueError(f"lp95/t no finitos: {lp95}, {t}")
        self._ema = lp95 if self._ema is None else (1 - self.cfg.latency_ema_alpha) * self._ema + self.cfg.latency_ema_alpha * lp95
        if self.baseline is None:
            self.baseline = self._ema
            return 0.0
        delta = (self._ema - self.baseline) / max(self.baseline, 1e-12)
        if (t - self._last_switch_t) < self.cfg.latency_refractory:
            return None
        return float(delta)

    @property
    def saturated(self) -> float | None:
        """Δ actual si excede el deadband, si no 0.0 (o None en refractario)."""
        if self._ema is None or self.baseline is None:
            return None
        delta = (self._ema - self.baseline) / max(self.baseline, 1e-12)
        return float(delta) if abs(delta) >= self.cfg.latency_deadband else 0.0
