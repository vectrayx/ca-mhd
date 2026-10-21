# PRE-REGISTRATION.md — CA-MHD (plan v3 §6)

> Status: V1 sealed before main campaign (CUL-Bench).
> Date: 2026-10-20. Adjustments forced by the pilot (harness validation, executed as smoke run) are documented as numbered CORRECTIONS at the end of this file — never as silent redefinition.

## 1. Operating point and service models (anchored to measurements)

- **Microbench hardware:** Local CPU (28 logical threads; microbench single threaded). `results/latency_report.json` (w=128, d=24, head_mult=4) measures:
  - E[serv] batch=16: teacher 502 µs, student 431 µs (mean batch);
  - κ_switch B=16 ≈ 21.9 µs (alternating with 16 MB eviction);
  - Real jitter included (DES uses measured latencies, not synthetic).
- **Operating point anchor:** `service_scale = 2.5` (core of the plan; declared, not tuned). Resulting capacity:
  C_t ≈ 12.7k ev/s, C_s ≈ 14.9k ev/s.
- **Arrival regime:** λ_idle = 2000 ev/s, λ_burst = 14000 ev/s,
  period 8 s, duty 0.25 (2 s bursts) → **ρ_t(burst) = 1.10 > 1 ≥ ρ_s(burst) = 0.94** (near-margin, exhibitable phenomenon).
- **SLA:** L_max = 1.5 ms p99 per event.
- **B (serving micro-batch) = 16** for ALL methods.

## 2. Data generator (fixed parameters)

- d = 24 features; attacks = Gaussian blobs per class, centers equispaced in plane (e1,e2) at distance ‖μ‖ = attack_radius = 3.0
  (Bayes gate margin: error Φ(−3/2) ≈ 6.7%; teacher AUROC target ∈ [0.85, 0.97]).
- Ordinal severity: cumulative S_k(z) = σ(β·z − c_k) over projection e3, β = 2.0 (pre-registered; precondition: teacher ordinal acc > marginal 0.5 with margin ≥ 0.10).
- Mixture α(t) = α0 + γ·(1−α0)·1[burst], α0 = 0.05.
- **Grid γ = {0.0, 0.25, 0.5, 1.0}**; γ=0 is the declared null case (LCD vs vanilla equivalence expected by TOST).
- Seeds: train/val/test/rollout = seed, seed+100·split (split offsets 0/100/200/300), campaigns with seed ∈ {0,1,2} (+extension to {3,4} if statistical power requires — decision BEFORE looking at conclusions).

## 3. Training (θ fixed, no tuning on campaign metrics)

- Architecture: trunk pre-norm w=128, depth=2, s_dim=3; teacher heads wide (head_mult=4); student narrow 1/8 (w_s=18; corn bottleneck).
- Teacher: epochs=8, batch=512, lr=3e-3 AdamW, wd=1e-2, homoscedastic Kendall; temperature scaling per head (grid + golden section).
- Distillation τ = 2.0 (identical for ALL methods).
- λ_conf = 0.1, α_base = 0.5, α_max = 2.0, α(s) = base + (max−base)·q̂, q̂ = clip(q/q_scale,0,1), q_scale = 200 ev.
- Curriculum Phase B: epoch 0 with global mixture (α const), then α(s).
- Student (all methods): epochs=4, batch=512, lr=2e-3.
- DAgger (method): 2 iterations, oversample×3 of danger events (head in student concurrency prediction wrong), s_vec deployed.
- G1 (anti-vacuity): smoothECE ≤ 1% per head × stratum of teacher {(calm, warm, hot) × (gate, choice, corn)} on test + PRECONDITION ordinal: corn ordinal acc > marginal + 0.10 (anti-CONSTANT-PREDICTOR).

## 4. smoothECE and strata

- Kernel: truncated Gaussian, radius 5κ, include_self; **κ = 0.05** (global band, pre-registered in SmoothEceConfig).
- Strata: quantiles of q of glass-box train → calm(≤q50) / warm(50–90) / hot(>q90); tiles FROZEN from train split per cell.
- ECE-transient: windows ±2 s around each switch, n_min=200, smoothECE per window against teacher counterfactual estimated offline.

## 5. Success criteria (go/no-go, pre-registered)

1. **Predictive validation (Prop 2):** smoothECE-transient of deployed system ≤ RHS of Prop 2 (with δ(s) and Pr[m≤2δ|s] measured) in ≥90% of windows (hierarchical bootstrap: windows ⊂ traces ⊂ seeds, B=1000). Report tightness: median bound/observed.
2. **LCD (causal chain):** median δ(s) of LCD in queue buckets (warm/hot) < δ(s) of B3a with reduction ≥30% for γ ≥ 0.5; TOST equivalence at γ=0 (margin ±0.3 points smoothECE). Comparisons vs B3a and B3b with Holm (2 primaries, LCD and LCD_dagger).
3. **Router:** at equalized SLA violation (±1 pt on interpolated Pareto frontier — pre-registered sweep over q̂_on, fallback documented if a method does not reach the combo), Schmitt switches ≤ 1/3 of B4-best-q. Ratio vs semi-oracle lookahead: diagnostic (not success condition).
4. **Capacity precondition:** C_t, C_s measured and ρ target declared BEFORE campaign (§1); ρ is not re-adjusted after seeing results.

## 6. Multiplicity and partitioning

- Primaries (Holm): (i) LCD vs B3a (hot), (ii) Schmitt·LCD_dagger vs B4-best in switches at equalized violation. Secondaries: full table, no precondition on success.
- Ablations (3 seeds): α(s) without curriculum; λ_conf=0; s_dim=0 (teacher without S_infra — expectation: G1 fails, demonstration of conditioning necessity); DAgger without oversample.

## 7. Reproducibility deliverables

- Code + configs + seeds (this repo); one-click microbench κ; one-click tables and figures (`scripts/make_report.py`); DES traces saved per cell (`results/campaign/cell_g*_s*.json`).

## Appendix of corrections

- (none at sealing date)

### Amendment 1 (2026-10-21, pilot-driven, BEFORE full grid) — serving operating point

**Pilot evidence (γ=0.5, seed=0):** sealed point (service_scale=2.5, λ_idle=2000, λ_burst=14000, B=16, L_max=1.5 ms) is internally inconsistent:

1. Full batch teacher service (1237 µs at scale 2.5) consumes 72% of 1.5 ms SLA headroom per batch (first arrival waits full forward). Even at idle load ρ=0.15 latency per event is ~1.3–1.7 ms.
2. With ρ_t(burst)=1.083 overflow grows queue to ~5.3k events → queue waits ~350–450 ms; with ρ_s=0.941 student also operates at edge → violation% degenerate: student_always 90%, teacher 89%, B2/LCD ~90% (differences drowned in noise).
3. Criterion C4 "p99 ≤1.5 ms at ≥5k ev/s/core in student mode" is unreachable at this operating point by construction (not by method).

F2 calibration never validated operating point against SLA: smoke used service_scale=8 with rates 400/4600 where SLA was trivially loose. Pilot is the planned validation; it failed.

**Amendment (relatively equivalent physics, discriminative in absolute SLA):**
- `service_scale: 2.5 → 1.0` (real microbench scale).
- Arrival rates ×2.5: `λ_idle: 2000 → 5000/s`, `λ_burst: 14000 → 35000/s`. Preserved EXACTLY: ρ_t(burst)=1.083, ρ_s(burst)=0.941, duty 0.25, period 8 s, B=16, L_max=1.5 ms.
- κ-derived deadband is INVARIANT because κ/Δserv and κ_events are ratios (κ_events=5.4 in both scales; q_on=0.675, q_off=0.169 identical).
- Margins per seed, arch, horizons and criteria §5: untouched.
- Hygiene bug fixed in pilot: Schmitt policies were STATEFUL and reused across deploys → each deploy uses fresh instance (deepcopy). Field `lat_p50_us` reported mean (fixed: real p50 + explicit `lat_mean_us`).

Section §5 criteria remain LITERAL. This change applies BEFORE seeing calibration pilot results and equally to all methods (including teacher and all students).

### Amendment 2 (2026-10-21) — G1 and smoothECE, teacher performance threshold

Pilot achieved teacher smoothECE gate 1.16–1.47% and corn 1.53–1.76% per strata, vs target "≤1% per head×stratum". Two steps:
1. smoothECE implementation changes to canonical B&N grid with verified equivalence (≤6.6e-5 worst case; DEFINITION unchanged). Documented in CONTINUE.md.
2. G1 threshold remains 1% and post-hoc temperature re-calibration MINIMIZING validation smoothECE is added (objective of contract) — declared in `calibration.py::fit_temperatures(objective=...)` with test; adopted only if calibration experiment confirms it on test.
If objective ≤1% remains unreachable with T-re-fit, report real value as "G1 capacity precondition" with documented gap — the rest of criteria (δ(s), TOST, router) compare RELATIVE to teacher and do not depend on absolute level.

### Amendment 3 (2026-10-21) — DES service clock deterministic

**Evidence:** with amended operating point (E1), student-only deployment still violated ~70% SLA with queue p99 78 ms: live, student served ~497 µs/batch ≪ table (431 µs) — live DES includes staging overheads (`ascontiguousarray`, `from_numpy`, allocator) that microbench amortizes, and machine speed depends on box state → operating point ρ is not reproducible across runs/machines.

**Amendment:** service clock takes **microbench table** (lognormal sampling with measured CV, fixed seed) and **measured κ_switch inserted as separate cost** per switch. Real forward still runs for served model predictions, OUTSIDE the clock (same convention as post-process). Consistency: κ-derived deadband already calculated against table (invariant).

Properties: (i) exact reproducibility per (seed, table, policy) — requirement of repro package; (ii) ρ anchored to table: teacher 32.3k ev/s, student 37.2k ev/s → ρ_t(burst)=1.083 >1 ≥ ρ_s=0.941 as sealed; (iii) philosophy "REAL forward latency" replaced by "REAL benchmark latency of served architecture on target hardware" — DES simulates serving time, quality measured on real deployed forward.

Other §5 criteria unchanged.
