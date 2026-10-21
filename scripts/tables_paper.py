"""Emitte las tablas LaTeX del paper desde los resultados de la campaña.

Salidas: paper/tables/{serving, g2, scorecard}.tex
  - serving.tex   : viol/p99/switches/cov por (γ, método×despliegue)
  - g2.tex        : δ(s) hot del grid (LCD/B3a/B3b + reducciones) por γ×head
  - scorecard.tex : criterios §5 go/no-go con su estado medido
Uso: PYTHONPATH=src .venv/bin/python scripts/tables_paper.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)

ROWS = [
    ("teacher", "teacher_always", "B1: teacher (saturado)"),
    ("LCD", "student_always", "B1': student-only (LCD)"),
    ("LCD_dagger", "schmitt", "Schmitt·LCD+DAgger (ours)"),
    ("LCD", "schmitt", "Schmitt·LCD (ours)"),
    ("LCD", "fixed_threshold", "B4: fixed-threshold (best-q)"),
    ("LCD", "random_matched", "B5: random matched"),
]


def load() -> list[dict]:
    rows = []
    for f in sorted(glob.glob("results/campaign/cell_g*.json")):
        rows.extend(json.load(open(f)))
    return rows


def emit_serving(rows: list[dict]) -> None:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"$\gamma$ & Método & viol\% & p99\,ms & switches & cov \\",
             r"\midrule"]
    for g in (0.0, 0.25, 0.5, 1.0):
        for m, pol, label in ROWS:
            sel = [r for r in rows if r["gamma"] == g and r["method"] == m
                   and r["policy"] == pol]
            if not sel:
                continue
            v = np.mean([r["sla_viol_rate"] * 100 for r in sel])
            sd = np.std([r["sla_viol_rate"] * 100 for r in sel])
            p99 = np.mean([r["lat_p99_us"] for r in sel]) / 1000
            sw = np.mean([r["n_switches"] for r in sel])
            cov = np.mean([r["student_coverage"] for r in sel])
            gtxt = f"$\\gamma{g:g}$" if (m, pol) == ROWS[0][0:2] else ""
            lines.append(
                f"{gtxt} & {label} & {v:.1f}$\\pm$({sd:.1f}) & {p99:.1f} & "
                f"{sw:,.0f} & {cov:.2f} \\\\")
        if g != 1.0:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "serving.tex").write_text("\n".join(lines) + "\n")


def emit_g2(rows: list[dict]) -> None:
    train = [r for r in rows if r["policy"] == "training"
             and "student_smoothECE" in r]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"$\gamma$ & head & LCD & B3a & B3b & LCD/B3a & LCD/B3b \\",
             r"\midrule"]
    for g in (0.0, 0.25, 0.5, 1.0):
        for head in ("gate", "choice", "corn"):
            med = {}
            for m in ("LCD", "B3a", "B3b"):
                xs = [r["student_smoothECE"]["hot"][head] * 100 for r in train
                      if r["gamma"] == g and r["method"] == m]
                med[m] = np.median(xs) if xs else float("nan")
            r1 = 100 * (1 - med["LCD"] / med["B3a"])
            r2 = 100 * (1 - med["LCD"] / med["B3b"])
            gtxt = f"$\\gamma{g:g}$" if head == "gate" else ""
            lines.append(
                f"{gtxt} & {head} & {med['LCD']:.2f} & {med['B3a']:.2f} & "
                f"{med['B3b']:.2f} & {r1:+.1f}\\% & {r2:+.1f}\\% \\\\")
        if g != 1.0:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "g2.tex").write_text("\n".join(lines) + "\n")


def emit_scorecard() -> None:
    rows = [
        ("T1 smoothECE-fit adoption (amend.\\ 2)",
         "gate/choice 1.3--1.5\\% vs meta 1\\%; corn $\\le$1.2\\%",
         "adoptado; brecha G1 estructural documentada"),
        ("T2a LCD vs B3a ($\\ge$30\\% hot)",
         "+4.7--9.3\\% corn $\\gg$ cabezas $\\pm$0--3\\%",
         "NO meta en cabezas; signo consistente (3/3 seeds)"),
        ("T2a' LCD vs B3b (ciego)",
         "corn 36--82\\%; cabezas 8--13\\%",
         "CONFIRMADO (B3b catastrófico en corn)"),
        ("T2b TOST $\\gamma{=}0$ ($\\pm$0.3 pts)",
         "gate/choice PASS; corn favorable-fallo",
         "PASS cabezas; corn mejor que B3a (3/3)"),
        ("T3 router ($\\le$1/3 B4, viol igualada)",
         "4--6 vs 200--2044 sw; $|\\Delta$viol$|\\!\\le$0.3 pts",
         "PASS todos los $\\gamma$ (sweep $q_{\\mathrm{on}}$)"),
        ("T4 transiente (Prop\\ 2)",
         "corn drenaje 5.8--17.8\\% vs $\\le$1.2\\% estát.",
         "medido; RHS formal en apéndice"),
        ("T5 precondición de capacidad ($\\rho$)",
         "anclado pre-campaña, sin re-ajuste",
         "PASS (Enmiendas 1--3 previsionadas)"),
    ]
    lines = [r"\begin{tabular}{p{5.2cm}p{5.4cm}p{5.0cm}}", r"\toprule",
             r"Criterio (§5) & Evidencia & Estado \\", r"\midrule"]
    for a, b, c in rows:
        lines.append(f"{a} & {b} & {c} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "scorecard.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load()
    emit_serving(rows)
    emit_g2(rows)
    emit_scorecard()
    for p in OUT.glob("*.tex"):
        print(f"→ {p}")


if __name__ == "__main__":
    main()
