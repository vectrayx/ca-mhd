"""Constantes del deadband κ-derivado (verificación NUMÉRICA, plan §3.4).

Lee results/latency_report.json (fuente de verdad del DES) y deriva
(q_on, q_off) exactamente como derive_thresholds_from_latency; imprime la
tabla de constantes con la verificación adversa deadband ≥ 1.5κ y la
equivalencia en presión. Sin tuneo: solo reporta.

Uso: PYTHONPATH=src .venv/bin/python scripts/derive_deadband.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from camhd.policies import derive_thresholds_from_latency  # noqa: E402

REPORT = Path("results/latency_report.json")
Q_REF = 16.0
# Enmienda 1 del pre-registro (pilot 2026-10-21): escala real del serving,
# tasas ×2.5 — la física relativa (ρ, deadband) es invariante a escala.
SERVICE_SCALE = 1.0
KAPPA_SCALE = 1.0
RATES = (("idle", 5000.0), ("burst", 35000.0))


def main() -> None:
    rep = json.loads(REPORT.read_text())
    th = derive_thresholds_from_latency(rep, q_ref=Q_REF,
                                        kappa_scale=KAPPA_SCALE,
                                        service_scale=SERVICE_SCALE)
    B = 16
    kappa = th["kappa_us"]
    delta_lote = th["delta_serv_us"]
    delta_event = delta_lote / B
    kappa_events = kappa / delta_event
    deadband_ev = th["q_on"] * Q_REF - th["q_off"] * Q_REF
    deadband_us = deadband_ev * delta_event

    print("report:", REPORT)
    print(f"  κ_switch(B16, ×{KAPPA_SCALE:g})      = {kappa:8.1f} µs")
    print(f"  E[serv_teacher lote T16]  = {th.get('E_serv_t_us', float('nan')):8.1f} µs" if "E_serv_t_us" in th else "  E[serv_teacher]           = (ver delta)")
    print(f"  Δserv lote (T−S ×{SERVICE_SCALE:g})    = {delta_lote:8.1f} µs")
    print(f"  Δserv por evento          = {delta_event:8.2f} µs")
    print(f"  κ_eventos = κ/Δserv_ev    = {kappa_events:8.2f}")
    print(f"  q_on  = min(1, max(0.30, 2κ_ev/q_ref)) = {th['q_on']:.3f} "
          f"(= {th['q_on'] * Q_REF:.1f} eventos)")
    print(f"  q_off                          = {th['q_off']:.3f} "
          f"(= {th['q_off'] * Q_REF:.1f} eventos)")
    print(f"  deadband = (q_on−q_off)·q_ref  = {deadband_ev:6.1f} eventos "
          f"↔ {deadband_us:6.1f} µs")
    ok1 = deadband_us >= 1.5 * kappa
    ok2 = th["q_on"] * Q_REF >= 2.0 * kappa_events
    print(f"  ASSERT deadband_us ≥ 1.5·κ : {'OK' if ok1 else 'FALLA'}")
    print(f"  ASSERT q_on ≥ 2·κ_eventos  : {'OK' if ok2 else 'FALLA'}")
    rho = None
    bt = rep["batch_table"]["teacher"]["16"]["mean"] * SERVICE_SCALE
    bs = rep["batch_table"]["student"]["16"]["mean"] * SERVICE_SCALE
    print(f"  E[serv] T16={bt:7.1f} µs  S16={bs:7.1f} µs")
    for name, lam in RATES:
        rho_t = lam * bt / B / 1e6
        rho_s = lam * bs / B / 1e6
        print(f"    ρ_t({name} λ={lam:g}) = {rho_t:.3f}   ρ_s = {rho_s:.3f}")
    if not (ok1 and ok2):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
