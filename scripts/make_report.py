"""Reporte one-click de la campaña (plan v3 §6): tablas + figuras F5.

Entrada: results/campaign/cell_g*_s*.json (JSONL por celda).
Salida: results/campaign/{summary.csv, fig1_gamma.png, fig2_pareto.png,
fig3_strata.png, fig4_dagger.png} + resumen impreso.

Los criterios G1/G2/G3 (§5 del plan) se reportan contra
PRE-REGISTRATION.md — este script NO decide, solo mide y dibuja.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMPAIGN_DIR = Path("results/campaign")

ROWS_T = {
    "teacher": "B1 (teacher, techo)",
    "B2_student_always": "B2 (student-only)",
    "B3a_student_always": "B3a (vanilla KD)",
    "B3b_student_always": "B3b (KD+oversampling)",
    "LCD_student_always": "LCD (α(s))",
    "LCD_schmitt": "Ours: Schmitt·LCD",
    "LCD_dagger_schmitt": "Ours: Schmitt·LCD+DAgger",
    "LCD_random_matched": "B5 (random matched)",
    "LCD_fixed_threshold": "B4 (best-q fixed)",
}


def _row_key(row: dict) -> str:
    return f"{row['method']}_{row['policy']}"


def load_cells(d: Path = CAMPAIGN_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(d.glob("cell_g*_s*.json")):
        for r in json.loads(p.read_text()):
            r = dict(r)
            key = _row_key(r)
            r["_key"] = key
            r["_label"] = ROWS_T.get(key, key)
            se = r.get("smoothECE") or {}
            for strata in ("calm", "warm", "hot", "all"):
                for head in ("gate", "choice", "corn"):
                    r[f"se_{strata}_{head}"] = se.get(strata, {}).get(head, np.nan)
            acc = r.get("accuracy") or {}
            for strata in ("calm", "warm", "hot"):
                r[f"acc_{strata}_gate"] = acc.get(strata, {}).get("gate", np.nan)
            # δ(s) := smoothECE del estrato − smoothECE del calm (shift relativo)
            for head in ("gate", "choice", "corn"):
                base = se.get("calm", {}).get(head, np.nan)
                hot = se.get("hot", {}).get(head, np.nan)
                r[f"delta_hot_{head}"] = hot - base if np.isfinite(base) and np.isfinite(hot) else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    return df


def worst_stratum(row) -> float:
    vals = [row.get(f"se_{s}_{h}", np.nan) for s in ("calm", "warm", "hot")
            for h in ("gate", "choice", "corn")]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else np.nan


def summary_table(df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    sub = df[(df["gamma"] == gamma) & df["_key"].isin(ROWS_T)].copy()
    sub["worst_se"] = sub.apply(worst_stratum, axis=1)
    agg = sub.groupby("_key").agg(
        label=("_label", "first"),
        worst_se=("worst_se", "median"),
        sla_viol_rate=("sla_viol_rate", "median"),
        n_switches=("n_switches", "median"),
        student_cov=("student_coverage", "median"),
        lat_p99=("lat_p99_us", "median"),
        auroc=("auroc_gate", "median"),
        n=("seed", "count"),
    ).sort_values("worst_se")
    return agg


def fig1_vs_gamma(df: pd.DataFrame, out: Path) -> None:
    """Fig 1 del plan: métricas vs γ por método (con barras de error seeds)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    core = df[df["_key"].isin(ROWS_T)]
    core["worst_se"] = core.apply(worst_stratum, axis=1)
    a = axes[0]
    for key, gsub in core.groupby("_key"):
        gm = gsub.groupby("gamma")["worst_se"].agg(["median", "std", "count"])
        a.errorbar(gm.index, gm["median"], yerr=gm["std"].fillna(0) / np.sqrt(gm["count"]),
                   marker="o", capsize=2, label=ROWS_T[key])
    a.set_xlabel("γ (acoplamiento carga↔contenido)")
    a.set_ylabel("smoothECE peor estrato (mediana)")
    a.set_title("a) Calibración desplegable vs γ")
    a.set_yscale("log")
    a.grid(alpha=0.3)
    b = axes[1]
    hot_keys = [k for k in ROWS_T if "student_always" in k or "schmitt" in k]
    hh = df[df["_key"].isin(hot_keys)]
    for key, gsub in hh.groupby("_key"):
        col = max(gsub.columns, key=str)
        gm = gsub.groupby("gamma")["delta_hot_gate"].agg(["median", "std", "count"])
        b.errorbar(gm.index, gm["median"] * 100,
                   yerr=gm["std"].fillna(0) / np.sqrt(gm["count"]) * 100,
                   marker="s", capsize=2, label=ROWS_T[key])
        b.set_xlabel("γ"); b.set_ylabel("δ(hot) del gate (pts)")
        b.set_title("b) Shift de calibración por carga (gate, hot)")
        b.grid(alpha=0.3)
    c = axes[2]
    for key, gsub in core.groupby("_key"):
        gm = gsub.groupby("gamma")["sla_viol_rate"].agg(["median", "std", "count"])
        c.errorbar(gm.index, gm["median"] * 100,
                   yerr=gm["std"].fillna(0) / np.sqrt(gm["count"]) * 100,
                   marker="^", capsize=2, label=ROWS_T[key])
        c.set_xlabel("γ"); c.set_ylabel("violación SLA (%)")
        c.set_title("c) Violación de SLA vs γ")
        c.set_yscale("symlog")
        c.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncols=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160)
    print("→", out)


def fig2_pareto(df: pd.DataFrame, gamma: float, out: Path) -> None:
    sub = df[(df["gamma"] == gamma) & df["_key"].isin(ROWS_T)].copy()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for key, gsub in sub.groupby("_key"):
        ax.scatter(gsub["n_switches"], gsub["sla_viol_rate"] * 100,
                   s=42, label=ROWS_T[key])
        med = gsub[["n_switches", "sla_viol_rate"]].median().to_numpy()
        ax.annotate(ROWS_T[key], (med[0], med[1] * 100 + 0.4), fontsize=7)
    ax.set_xlabel("conmutaciones (κ facturado)")
    ax.set_ylabel("violación SLA (%)")
    ax.set_title(f"Pareto de serving (γ={gamma:g}): Rossberg vs B4/B5")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("→", out)


def fig3_strata(df: pd.DataFrame, gamma: float, out: Path) -> None:
    lcd = df[(df["gamma"] == gamma) & (df["_key"] == "LCD_student_always")]
    b3a = df[(df["gamma"] == gamma) & (df["_key"] == "B3a_student_always")]
    strata = ("calm", "warm", "hot")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    w = 0.35
    xs = np.arange(len(strata))
    for i, (dff, lab, col) in enumerate(((lcd, "LCD (α(s))", "C2"),
                                          (b3a, "B3a vanilla KD", "C3"))):
        means = [dff[f"se_{s}_gate"].median() * 100 for s in strata]
        ax.bar(xs + (i - 0.5) * w, means, w, label=lab, color=col)
    ax.set_xticks(xs, strata)
    ax.set_ylabel("smoothECE gate por estrato (%)")
    ax.set_title(f"δ(s) condicional a carga (γ={gamma:g})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("→", out)


def main(gamma_main: float = 0.5) -> None:
    df = load_cells()
    if df.empty:
        print("sin celdas aún en", CAMPAIGN_DIR)
        return
    out = Path(CAMPAIGN_DIR)
    df.to_csv(out / "summary.csv", index=False)
    print(df[["_key", "gamma", "seed", "sla_viol_rate"]].to_string(index=False))
    sm = summary_table(df, gamma_main)
    sm.to_csv(out / f"summary_g{gamma_main:g}.csv")
    print("\n=== tabla principal (γ = %g) ===\n" % gamma_main)
    print(sm.to_string())
    fig1_vs_gamma(df, out / "fig1_gamma.png")
    fig2_pareto(df, gamma_main, out / "fig2_pareto.png")
    fig3_strata(df, gamma_main, out / "fig3_strata.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.5)
    a = ap.parse_args()
    main(a.gamma)
