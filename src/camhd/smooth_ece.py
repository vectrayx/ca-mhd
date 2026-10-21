"""smoothECE (Blasiok & Nakkiran, ICLR 2024) y ECE binned, implementación propia.

Elecciones documentadas (CONTINUE.md §5, con nota de optimización v4):

- Suavización RBF gaussiana en el espacio de confianza top-label, con la
  CURVA DE FIT f̄(c) ≈ P(correct | top-label confidence = c) y
  smoothECE = mean_i |c_i − f̄(c_i)| (estimador muestral, no el integral
  sobre grilla de relplot; la definición pre-registrada es esta).

- Para n grandes el cálculo se hace con el algoritmo CANÓNICO de B&N
  (Apple relplot, kernels.py): scatter de las observaciones sobre una grilla
  uniforme de E = max(2000, 20/κ) puntos con split lineal por bin, carrusel
  discrete-convolve con el kernel gaussiano en la grilla, e interpolación
  lineal de la curva en cada c_i observado. Coste O(n + E²) — con κ=0.05,
  E=2000 → milisegundos en 255k puntos. Error de redondeo de la grilla
  O(h²·|φ''|) con h=5e-4 → despreciable frente al ruido muestral y los
  tamaños de efecto del estudio (~1e-2). Verificado con test de regresión
  contra el cálculo pairwise exacto en datos tricolores saturados.

- Para n ≤ _EXACT_LIMIT se conserva el cálculo pairwise EXACTO por punto
  (ventana searchsorted ±5κ, peso del propio punto incluido); también se
  usa como salida de 参考 en brute-force tests.

- `include_self=False` queda disponible como ablation (excluye el peso
  propio en la ventana).

Ambas métricas asumen top-label: `confidence` = probabilidad de la clase
ganadora ∈ [0,1] y `correctness` ∈ {0,1} (o pesos en [0,1] para muestras
ponderadas, p. ej. en estratos de carga).
"""

from __future__ import annotations

import numpy as np

__all__ = ["smooth_ece", "binned_ece", "smoothed_correctness_curve"]

_EXACT_LIMIT = 4096


def _sorted_confidence(
    confidence: np.ndarray, correctness: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidence, dtype=np.float64).ravel()
    y = np.asarray(correctness, dtype=np.float64).ravel()
    if c.shape != y.shape:
        raise ValueError(f"confidence y correctness deben tener igual forma: {c.shape} vs {y.shape}")
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(y)):
        raise ValueError("confidence/correctness contienen NaN/inf")
    if np.any(c < 0) or np.any(c > 1):
        raise ValueError("confidence fuera de [0,1]")
    order = np.argsort(c, kind="stable")
    return c[order], y[order]


def _scatter_grid(c: np.ndarray, y: np.ndarray, eval_points: int) -> np.ndarray:
    """Distribuye masa de y sobre grilla uniforme con split lineal (canónico B&N).

    values[k] es la masa de y que encontró la tarea a la posición s_k = k/(E−1).
    """
    bins = (c * (eval_points - 1)).astype(np.int64).clip(0, eval_points - 2)
    frac = c * (eval_points - 1) - bins
    values = np.zeros(eval_points, dtype=np.float64)
    values += np.bincount(bins, weights=(1.0 - frac) * y, minlength=eval_points)
    values += np.bincount(bins + 1, weights=frac * y, minlength=eval_points)
    return values


def _kernel_curve_grid(c: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """f̄(c_i) via grilla + convolución discreta + interpolación lineal."""
    n = c.size
    eval_points = max(2000, int(round(20.0 / bandwidth)))
    values = _scatter_grid(c, y, eval_points)
    var = bandwidth * bandwidth
    t = np.linspace(0.0, 1.0, eval_points)
    ker = np.exp(-((t - 0.5) ** 2) / (2.0 * var))
    ker /= np.sqrt(2.0 * np.pi) * bandwidth
    smoothed = np.convolve(values, ker, mode="same")
    dens = np.convolve(_scatter_grid(c, np.ones_like(y), eval_points), ker, mode="same")
    denom = np.where(dens > 0.0, dens, 1.0)
    ratio = smoothed / denom  # curva de fit en la grilla
    # interpolación lineal en cada c_i observado
    pos = c * (eval_points - 1)
    i0 = np.clip(pos.astype(np.int64), 0, eval_points - 2)
    w = pos - i0
    return ratio[i0] * (1.0 - w) + ratio[i0 + 1] * w


def _curve_exact(c: np.ndarray, y: np.ndarray, bandwidth: float, radius_scale: float, include_self: bool) -> np.ndarray:
    """Cálculo pairwise exacto por punto (ventana ±5κ searchsorted)."""
    n = c.size
    radius = bandwidth * radius_scale
    out = np.empty(n, dtype=np.float64)
    y_global = float(y.mean())
    lo_all = np.searchsorted(c, c - radius, side="left")
    hi_all = np.searchsorted(c, c + radius, side="right")
    for i in range(n):
        ci = c[i]
        lo, hi = int(lo_all[i]), int(hi_all[i])
        if include_self and hi - lo == 1:
            # ventana = solo el punto propio → fit = y_i (límite κ→0 exacto)
            out[i] = y[i]
            continue
        # pesos recalculados SIEMPRE contra el centro actual (ci):
        # un enfoque de acumulador arrastrado congela el peso que el punto
        # tenía al ENTRAR por el borde de la ventana (≈0) — bug v1 detectado
        # con brute-force comparison y corregido aquí.
        w = np.exp(-0.5 * ((c[lo:hi] - ci) / bandwidth) ** 2)
        if not include_self and lo <= i < hi:
            w[i - lo] = 0.0
        den = float(w.sum())
        out[i] = float(w @ y[lo:hi]) / den if den > 0.0 else y_global
    return out


def smoothed_correctness_curve(
    confidence: np.ndarray,
    correctness: np.ndarray,
    bandwidth: float = 0.10,
    radius_scale: float = 5.0,
    include_self: bool = True,
) -> np.ndarray:
    """Retorna f̄(c_i) = Σ_j K((c_i−c_j)/κ) y_j / Σ_j K((c_i−c_j)/κ).

    La CURVA DE FIT (reliability diagram suavizado de B&N): f̄(c) ≈ P(correct
    | top-label confidence = c) en los puntos observados. smoothECE =
    mean_i |c_i − f̄(c_i)|.

    n ≤ _EXACT_LIMIT → pairwise exacto; n mayor → algoritmo canónico de
    grilla de B&N (ver docstring del módulo). Los tests de regresión
    comparan ambas rutas con tolerancia de grilla.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth debe ser > 0, got {bandwidth}")
    c, y = _sorted_confidence(confidence, correctness)
    n = c.size
    if n == 0:
        raise ValueError("sin muestras")
    if n <= _EXACT_LIMIT:
        return _curve_exact(c, y, bandwidth, radius_scale, include_self)
    return _kernel_curve_grid(c, y, bandwidth)


def smooth_ece(
    confidence: np.ndarray,
    correctness: np.ndarray,
    bandwidth: float = 0.10,
    radius_scale: float = 5.0,
    include_self: bool = True,
) -> float:
    """smoothECE ≡ mean_i |c_i − f̄_κ(c_i)| con kernel RBF de ancho κ.

    Con κ→0 converge al error local (estimador sobre-fiteado); con κ→∞ al
    |c_i − accuracy|. κ es una cantidad pre-registrada del plan v3.
    """
    curve = smoothed_correctness_curve(
        confidence, correctness, bandwidth=bandwidth, radius_scale=radius_scale, include_self=include_self
    )
    c = np.sort(np.asarray(confidence, dtype=np.float64).ravel())
    return float(np.abs(c - curve).mean())


def binned_ece(confidence: np.ndarray, correctness: np.ndarray, n_bins: float = 15) -> float:
    """ECE binned clásico (para comparabilidad con la literatura; la cota del
    plan v3 se enuncia SOLO sobre smoothECE porque el binned es discontinuo).
    """
    c, y = _sorted_confidence(confidence, correctness)
    if c.size == 0:
        raise ValueError("sin muestras")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, int(n_bins) - 1)
    n = c.size
    ece = 0.0
    for b in range(int(n_bins)):
        mask = idx == b
        m = int(mask.sum())
        if m == 0:
            continue
        ece += (m / n) * abs(float(y[mask].mean()) - float(c[mask].mean()))
    return float(ece)
