"""Figura 4 del paper: transiente ECE-swap por ventanas pre/post-switch.

Entrada: results/f5/cell_g*.json (scripts/f5_transient.py).
Salida : results/campaign/fig4_transient.png + fig4_transient.pdf

Diseño: 3 paneles (γ = 0, 0.5, 1.0); para cada switch de cada seed:
ΔsECE = post − pre (pts) por cabeza (gate, choice, corn) con IC 90%
boot (jerárquico por bloques — ya calculado en el JSON de F5).
Línea 0 = régimen estacionario; el corolario del paper: el corn en las
ventanas cortas de drenaje muestra los mayores valores (~+5 a +9 pts).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/campaign/fig4_transient.png")


def main() -> None:
    data = []
    for f in sorted(glob.glob("results/f5/cell_g*.json")):
        data.append(json.load(open(f)))
    gammas = sorted({d["gamma"] for d in data})
    heads = ("gate", "choice", "corn")
    colors = {"gate": "#1f77b4", "choice": "#2ca02c", "corn": "#d62728"}

    fig, axes = plt.subplots(1, len(gammas), figsize=(4.2 * len(gammas), 3.6),
                             sharey=True)
    for ax, g in zip(axes, gammas):
        cells = [d for d in data if d["gamma"] == g]
        i = 0
        for ci, d in enumerate(cells):
            for w in d["windows"]:
                for h in heads:
                    pre, post = w[h]["pre"], w[h]["post"]
                    if not (np.isfinite(pre["se"]) and np.isfinite(post["se"])):
                        continue
                    if pre["n"] < 200 or post["n"] < 200:
                        continue
                    # error IC90 del Δ con conservadurismo de mitades
                    e_pre = (pre["ci"][1] - pre["ci"][0]) / 2
                    e_post = (post["ci"][1] - post["ci"][0]) / 2
                    lo = (post["se"] - e_post) - (pre["se"] + e_pre)
                    hi = (post["se"] + e_post) - (pre["se"] - e_pre)
                    dlt = (post["se"] - pre["se"]) * 100
                    x = i + (heads.index(h) - 1) * 0.3
                    ax.errorbar(x, dlt,
                                yerr=[[max(0.0, dlt - lo * 100)],
                                      [max(0.0, hi * 100 - dlt)]],
                                fmt="o", ms=3, color=colors[h], alpha=0.75,
                                lw=1, capsize=1.5)
                i += 1
        for h in heads:  # leyenda simulada con puntos invisibles
            ax.plot([], [], "o", color=colors[h], label=h)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"$\\gamma$ = {g}")
        ax.set_xlabel("ventanas de switch (seeds apiladas)")
    axes[0].set_ylabel(r"$\Delta$sECE post$-$pre  (pts, $\times100$)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Transiente de calibración por conmutación "
                 r"(ventanas ±2 s, IC90$^{*}$ boot; $n\geq200$)", y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
