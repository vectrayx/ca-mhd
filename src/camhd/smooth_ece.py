"""smoothECE (Blasiok & Nakkiran, ICLR 2024) and ECE binned, own implementation.

Documented choices (CONTINUE.md §5, with optimization note v4):

- Gaussian RBF smoothing in the top-label trust space, with the
FIT CURVE f (c) ≈ P(correct | top-label confidence = c) and
smoothECE = mean_i |c_i − f － (c_i)| (sample estimator, not integral
on relplot grid; the pre-recorded definition is this).

- For large n, the calculation is made with B&N's CANONICAL ALGORITHM
(Apple relplot, kernels.py): scatter of the observations on a grid
uniform of E = max(2000, 20/κ) points with linear split per bin, carousel
discrete-convolves with the Gaussian kernel in the grid, and interpolation
of the curve at each observed c_i. Cost O(n + E²) — with κ=0.05,
E=2000 → milliseconds at 255k points. Grid Rounding Error
O(h²·|φ''|) with h= → 5e-4 negligible versus sample noise and
study effect sizes (~1e-2). Verified with regression test
against exact pairwise calculation on saturated tricolor data.

- For n ≤ _EXACT_LIMIT the EXACT pairwise calculation per point is preserved
(searchsorted window ± 5κ, weight of the point itself included); it is also
used as output from 参考 in brute-force tests.

- `include_self=False` is available as ablation (excludes weight
own in the window).

Both metrics assume top-label: `confidence` = probability of the class
winner ∈ [0,1] and `correctness` ∈ {0,1} (or weights in [0,1] for samples
weighted, e.g. in load strata).
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
    """Distribute mass of and on a uniform grid with linear split (canonical B&W).

values[k] is the mass of y that found the task at position s_k = k/(E−1).
    """
    bins = (c * (eval_points - 1)).astype(np.int64).clip(0, eval_points - 2)
    frac = c * (eval_points - 1) - bins
    values = np.zeros(eval_points, dtype=np.float64)
    values += np.bincount(bins, weights=(1.0 - frac) * y, minlength=eval_points)
    values += np.bincount(bins + 1, weights=frac * y, minlength=eval_points)
    return values


def _kernel_curve_grid(c: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """f(c_i) via grid + discrete convolution + linear interpolation."""
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
    ratio = smoothed / denom  # fit curve on the grid
    # linear interpolation at each observed c_i
    pos = c * (eval_points - 1)
    i0 = np.clip(pos.astype(np.int64), 0, eval_points - 2)
    w = pos - i0
    return ratio[i0] * (1.0 - w) + ratio[i0 + 1] * w


def _curve_exact(c: np.ndarray, y: np.ndarray, bandwidth: float, radius_scale: float, include_self: bool) -> np.ndarray:
    """Exact pairwise calculation per point (±5κ searchsorted window)."""
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
            # window = only the proper point → fit = y_i (exact limit κ→0)
            out[i] = y[i]
            continue
        # recalculated weights ALWAYS against the current center (ci):
        # a dragged accumulator approach freezes the weight that the point
        # had when ENTERING through the edge of the window (≈0) — bug v1 detected
        # with brute-force comparison and corrected here.
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
    """Returns f (c_i) = Σ_j K((c_i−c_j)/κ) y_j / Σ_j K((c_i−c_j)/κ).

The FIT (B&N Smoothed Reliability Diagram) CURVE: f ̂ (c) ≈ P(correct
| top-label confidence = c) in the observed points. smoothECE =
mean_i |c_i − f ̂ (c_i)|.

n ≤ _EXACT_LIMIT → pairwise exact; n largest canonical → algorithm of
b&N grid (see module docstring). Regression tests
compare both routes with grid tolerance.
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
    """smoothECE ≡ mean_i |c_i − f －_κ(c_i)| with κ-width RBF kernel.

With κ→0 converges to the local error (over-phased estimator); with κ→∞ to
|c_i − accuracy|. κ is a pre-registered amount from plan v3.
    """
    curve = smoothed_correctness_curve(
        confidence, correctness, bandwidth=bandwidth, radius_scale=radius_scale, include_self=include_self
    )
    c = np.sort(np.asarray(confidence, dtype=np.float64).ravel())
    return float(np.abs(c - curve).mean())


def binned_ece(confidence: np.ndarray, correctness: np.ndarray, n_bins: float = 15) -> float:
    """Classic binned ECE (for comparability with the literature; the level of
plan v3 is stated ONLY on smoothECE because the binned is discontinuous).
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
