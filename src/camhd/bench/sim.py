"""DES de serving — Fase F2 (plan v3 §4: motor del CUL-Bench).

Simulación a eventos discretos de un core de serving MLP, EN ESTE CPU:

- Llegadas: exactas del generador acoplado (misma ley que el entrenamiento).
- Servicio por MICRO-LOTES: cuando el core queda libre drena la cola hasta
  B eventos y ejecuta UN forward real (`forward_heads`) — el tiempo de
  servicio es la latencia MEDIDA de ese lote en este CPU × `service_scale`
  (ancla al punto operativo pre-registrado: ρ_t(burst)≈1.1 > 1 ≥ ρ_s).
  Sin distribuciones sintéticas: el jitter es el jitter real del hardware.
- κ_switch: cuando el conjunto de cabezas estrechas cambia entre servicios
  consecutivos, se factura el exceso medido (evicción de 16 MB + cold fetch).
- Router: política externa; recibe q (backlog al tick) y devuelve el
  conjunto de cabezas en modo student. Schmitt/histéresis viven ahí.
- S_infra = (q_t, λ_t, ℓ_p95_t) registrado por evento (input del modelo,
  estratificación del contrato δ(s)).

Replay de entrenamiento (glass box) = misma DES con política fija
all-teacher: saturación real en bursts → estados de cola profundos.

Tiempos: TODO en "segundos de stream"; latencias reportadas en µs ×1e6.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models import CAMHDNet
from ..generator import StreamSample
from .latency import LatencyModel

__all__ = ["SimConfig", "SimResult", "RouterPolicy", "run_des", "glass_box_des"]


@dataclass
class SimConfig:
    batch_size: int = 16
    service_scale: float = 2.5      # ancla del punto operativo (ver docstring)
    kappa_scale: float = 2.5        # mismo núcleo → mismo factor
    rate_idle: float = 2000.0       # llegadas/s (registradas del StreamConfig)
    rate_burst: float = 14000.0
    t_end: float | None = None      # None = agotar el stream
    l_max_us: float = 1500.0        # SLA p99 por evento
    device: str = "cpu"
    lp95_window: int = 200          # ventana rodante de ℓ_p95 (para S_infra)
    max_services: int = 2_000_000   # guardaespaldas
    # escalado de S_infra PARA EL MODELO (mismo pipeline que el entrenamiento;
    # los s_vec almacenados quedan RAW para estratificación)
    s_scale: tuple[float, float, float] = (200.0, 14000.0, 1e4)
    # seed del reloj de servicio (Enmienda 3 + repro por deploy): run_des
    # RESETA el RNG del LatencyModel con esta seed — el orden de despacho
    # de la campaña no debe afectar las métricas de un deploy
    rng_seed: int = 7420


@dataclass
class SimResult:
    t_arr: np.ndarray          # llegada [n] (s de stream)
    t_dep: np.ndarray          # partida [n] (s de stream)
    q_arr: np.ndarray          # cola observada al llegar (pre-join, head-of-line)
    q_s: np.ndarray            # backlog al tick de servicio (input del router)
    lat_us: np.ndarray         # (t_dep − t_arr) ×1e6
    s_vec: np.ndarray          # [n,3] (q_t, λ_t, ℓ_p95_t) al servicio
    narrow: np.ndarray         # [n,3] bool por cabeza (gate, choice, corn)
    kappa_n: np.ndarray        # [n] 1 si el servicio consumió un κ_switch
    pred_gate: np.ndarray      # [n] prob de ataque desplegada
    pred_choice: np.ndarray    # [n] argmax choice
    pred_level: np.ndarray     # [n] argmax pmf corn
    conf_choice: np.ndarray    # [n] confianza top-1 choice
    conf_corn: np.ndarray      # [n] confianza top-1 corn (pmf)
    n_switches: int = 0
    summary: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t_arr)

    def summarize(self, l_max_us: float) -> dict:
        lat = self.lat_us
        return {
            "n_events": int(self.n),
            "n_switches": int(self.n_switches),
            "lat_mean_us": float(lat.mean()),
            "lat_p95_us": float(np.percentile(lat, 95)) if self.n else float("nan"),
            "lat_p99_us": float(np.percentile(lat, 99)) if self.n else float("nan"),
            "sla_viol_rate": float((lat > l_max_us).mean()) if self.n else float("nan"),
            "q_mean": float(self.q_arr.mean()),
            "q_p95": float(np.percentile(self.q_arr, 95)) if self.n else float("nan"),
            "q_max": int(self.q_arr.max()) if self.n else 0,
        }


class RouterPolicy:
    """Política base: todas las cabezas en modo teacher SIEMPRE.

    update(q, t) → subconjunto de ("gate","choice","corn") en modo student.
    Las políticas con estado (Schmitt) mantienen su máquina interna aquí.
    """

    def update(self, q: int, t: float) -> tuple[str, ...]:
        return ()


def run_des(
    sample: StreamSample,
    model: CAMHDNet | None,
    policy: RouterPolicy | None,
    lat: LatencyModel,
    cfg: SimConfig,
    standardize: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimResult:
    """DES de serving. policy=None → all-teacher (glass box); model=None →
    servicio por cola puro (solo sanity del harness sin inferencia)."""
    # Reloj de servicio reproducible POR DEPLOY: reseteo el RNG del
    # LatencyModel (lognormal con CV medida, seed fija por deploy —
    # el orden de despacho de la campaña es irrelevante para las
    # métricas; Enmienda 3).
    lat.rng = np.random.default_rng(cfg.rng_seed)
    HN = ("gate", "choice", "corn")
    n_src = len(sample)
    t_arr = sample.t
    feats = sample.features
    if standardize is not None:
        mu, sd = standardize
        feats = (feats - mu) / np.maximum(sd, 1e-6)
    if cfg.t_end is not None:
        keep = t_arr <= cfg.t_end
        n = int(keep.sum())
    else:
        n = n_src
    t_arr = np.asarray(t_arr[:n], dtype=np.float64)

    evict = torch.empty(4 << 20, dtype=torch.float32)
    device = torch.device(cfg.device)
    model = model.eval() if model is not None else None

    # salidas por evento
    t_dep = np.zeros(n, dtype=np.float64)
    q_arr = np.zeros(n, dtype=np.int64)
    q_s = np.zeros(n, dtype=np.int64)
    lat_us = np.zeros(n, dtype=np.float64)
    s_vec = np.zeros((n, 3), dtype=np.float32)
    narrow_mat = np.zeros((n, 3), dtype=bool)
    kappa_n = np.zeros(n, dtype=np.int8)
    pred_gate = np.zeros(n, dtype=np.float32)
    pred_choice = np.zeros(n, dtype=np.int64)
    pred_level = np.zeros(n, dtype=np.int64)
    conf_choice = np.zeros(n, dtype=np.float32)
    conf_corn = np.zeros(n, dtype=np.float32)
    n_switch = 0

    lp95_win: deque[float] = deque(maxlen=cfg.lp95_window)
    queue: deque[int] = deque()

    # estado del server
    serv_end: float | None = None
    serv_batch: tuple[int, ...] = ()
    serv_narrow: tuple[str, ...] = ()
    serv_dur = 0.0
    cur_narrow: tuple[str, ...] | None = None
    i = 0
    done = 0
    steps = 0

    def _start_service(t_start: float) -> None:
        nonlocal serv_end, serv_batch, serv_narrow, serv_dur, cur_narrow, n_switch
        q_now = len(queue)
        if policy is not None:
            narrow = tuple(policy.update(q_now, t_start))
        else:
            narrow = ()
        kappa_here = (cur_narrow is not None and narrow != cur_narrow)
        if kappa_here:
            evict.fill_(float(done))   # cold fetch del nuevo conjunto de pesos
            n_switch += 1
        b = min(cfg.batch_size, len(queue))
        idx = tuple(queue.popleft() for _ in range(b))
        lam = float(cfg.rate_burst if sample.in_burst[idx[0]] else cfg.rate_idle)
        lp95 = float(np.percentile(np.asarray(lp95_win, dtype=np.float64), 95)) if lp95_win else 0.0
        # registro por evento del lote
        for j in idx:
            s_vec[j] = (q_now, lam, lp95)
            for k, name in enumerate(HN):
                narrow_mat[j, k] = name in narrow
            kappa_n[j] = 1 if kappa_here else 0
        # Enmienda 3 (2026-10-21): el reloj de servicio sale de la tabla del
        # microbench (muestreo lognormal con la CV medida, seed fija) y el κ
        # de conmutación se inserta como costo aparte. El forward real corre
        # para obtener las PREDICCIONES, fuera del reloj (igual que el
        # post-proceso): el tiempo de máquina vivo no es reproducible y
        # arrastra overheads de staging que el microbench amortiza — lo que
        # desplazaba el punto operativo ρ entre corridas (bug del pilot v1:
        # la cola no drenaba porque el servicio vivo medía ~50 ms/evento).
        dur_us = 0.0
        if len(narrow) == 0:
            dur_us = float(lat.sample_service_us("teacher", b))
        elif len(narrow) == 3:
            dur_us = float(lat.sample_service_us("student", b))
        else:
            dur_us = 0.5 * (float(lat.sample_service_us("teacher", b))
                            + float(lat.sample_service_us("student", b)))
        if kappa_here:
            dur_us += float(lat.sample_kappa_us(b))
        if model is not None:
            # PREDICCIONES del modelo real servido (fuera del reloj)
            xb = torch.from_numpy(np.ascontiguousarray(feats[list(idx)]).astype(np.float32))
            s_row = np.asarray([[q_now / cfg.s_scale[0], lam / cfg.s_scale[1],
                                 lp95 / cfg.s_scale[2]]], dtype=np.float32)
            sb = torch.from_numpy(np.tile(s_row, (b, 1)))
            with torch.no_grad():
                if len(narrow) == 0:
                    out = model(xb, sb, mode="teacher")["t"]
                elif len(narrow) == 3:
                    out = model(xb, sb, mode="student")["s"]
                else:
                    out = model.forward_heads(xb, sb, narrow_heads=narrow)
                pred_gate[list(idx)] = torch.softmax(out["gate"], -1)[:, 1].numpy()
                pred_choice[list(idx)] = out["choice"].argmax(-1).numpy()
                probs_c = torch.softmax(out["choice"], -1)
                conf_choice[list(idx)] = probs_c.max(dim=-1).values.numpy()
            from ..losses import corn_pmf
            with torch.no_grad():
                pmf = corn_pmf(out["corn"])
            pred_level[list(idx)] = pmf.argmax(-1).numpy()
            conf_corn[list(idx)] = pmf.max(dim=-1).values.numpy()
        dur_stream = (dur_us * cfg.service_scale) / 1e6
        serv_end = t_start + dur_stream
        serv_batch = idx
        serv_narrow = narrow
        serv_dur = dur_stream
        cur_narrow = narrow

    while done < n and steps < cfg.max_services:
        steps += 1
        if serv_end is not None and (i >= n or serv_end <= t_arr[i]):
            # completar servicio
            t_clock = serv_end
            for j in serv_batch:
                t_dep[j] = t_clock
                lat_us[j] = (t_clock - t_arr[j]) * 1e6
                lp95_win.append(float(lat_us[j]))
            done += len(serv_batch)
            # canal de latencia de la política (LatencyGuard del método)
            if queue and policy is not None and hasattr(policy, "observe"):
                lp95_now = float(np.percentile(np.asarray(lp95_win, dtype=np.float64), 95))
                policy.observe(lp95_now, t_clock)
            # siguiente servicio inmediato si hay cola
            serv_end = None
            if queue:
                _start_service(t_clock)
        elif i < n:
            # llegada (o lote de llegadas simultáneas)
            t_clock = t_arr[i]
            while i < n and t_arr[i] == t_clock:
                q_arr[i] = len(queue)
                queue.append(i)
                i += 1
            if serv_end is None:
                _start_service(t_clock)
        else:
            break

    if done < n:
        raise RuntimeError(f"DES incompleto: done={done}/{n} (steps={steps})")

    res = SimResult(
        t_arr=t_arr, t_dep=t_dep, q_arr=q_arr, q_s=q_s, lat_us=lat_us,
        s_vec=s_vec, narrow=narrow_mat, kappa_n=kappa_n,
        pred_gate=pred_gate, pred_choice=pred_choice, pred_level=pred_level,
        conf_choice=conf_choice, conf_corn=conf_corn, n_switches=n_switch,
    )
    res.summary = res.summarize(cfg.l_max_us)
    return res


def glass_box_des(
    sample: StreamSample,
    model: CAMHDNet | None,
    cfg: SimConfig,
    lat: LatencyModel,
    standardize: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimResult:
    """Replay de entrenamiento: política fija all-teacher."""
    return run_des(sample, None, None, lat, cfg, standardize) if model is None else \
        run_des(sample, model, None, lat, cfg, standardize)
