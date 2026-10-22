"""Router with hysteresis (Schmitt trigger) + latency channel governance.

Plan design v3 §3.4 (G4 corrections):
- REUSABLE head automaton with dead zone [q_off, q_on).
- `min_dwell`: minimum time in mode after switching (residual anti-flapping
that the dead zone does not cover: q oscillating above q_on with peaks not
must toggle and toggle in milliseconds).
- `LatencyGuard`: ℓ_p95 rolling with EMA + minimum shift deadband +
refractory period after each switch — protects against looping
oscillator: latency → spike switch (κ real) → ℓ_p95 triggers → switch
Back
- Online cost is only observable (SLA violation + switching), Prop 3
v3: the one recorded here is exactly that accounting signal.
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
    """TEACHER→STUDENT Automaton with double hysteresis and minimal dwell."""

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
        """Evaluates the trigger with the current queue depth. Returns True
if switched (the caller must account for the κ_switch cost).
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
        """Temporal fraction in student mode from t=0."""
        if not self.records:
            return 0.0
        end = t_end if t_end is not None else self.records[-1].t
        if end <= 0:
            return 0.0
        total = 0.0
        # traverse intervals: student from switch-to-student t to
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
    """digsLatency channel condition: EMA of ℓ_p95 + deadband +
refractory period after switching. Does not make decisions — delivery
`saturated_signal` that the调用 router can use to block actions
driven by latency while the peak by κ_switch is metabolized.
    """

    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg
        self._ema: float | None = None
        self._last_switch_t: float = float("-inf")
        self.baseline: float | None = None  # Reference ℓ (stable regime)

    def on_switch(self, t: float) -> None:
        self._last_switch_t = t

    def feeding(self, lp95: float, t: float) -> float | None:
        """Updates rolling signal; returns normalized Δ against baseline
(0 if no baseline yet). None if the channel is in refractory.
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
        """Current Δ if it exceeds the deadband, if not 0.0 (or None in refractory)."""
        if self._ema is None or self.baseline is None:
            return None
        delta = (self._ema - self.baseline) / max(self.baseline, 1e-12)
        return float(delta) if abs(delta) >= self.cfg.latency_deadband else 0.0
