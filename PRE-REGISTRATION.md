# PRE-REGISTRATION.md — CA-MHD (plan v3 §6)

> Estado: V1 sellado antes de la campaña principal (CUL-Bench).
> Fecha: 2026-10-20. Los ajustes que el pilot (validación de harness,
> ejecutada como corrida de humo) fuerce se documentan como CORRECCIÓN
> numerada al final de este archivo — nunca como redefinición silenciosa.

## 1. Punto operativo y modelos de servicio (anclados a mediciones)

- **Hardware de microbench:** CPU local (28 hilos lógicos; microbench a
  1 hilo). `results/latency_report.json` (w=128, d=24, head_mult=4) mide:
  - E[serv] batch=16: teacher 502 µs, student 431 µs (media batch);
  - κ_switch B=16 ≈ 21.9 µs (par alternado con evicción de 16 MB);
  - jitter real incluido (el DES usa latencias medidas, no sintéticas).
- **Ancla del punto operativo:** `service_scale = 2.5` (núcleo objetivo
  del plan; se declara, no se tunea). Capacidad resultante:
  C_t ≈ 12.7k ev/s, C_s ≈ 14.9k ev/s.
- **Régimen de llegada:** λ_idle = 2000 ev/s, λ_burst = 14000 ev/s,
  periodo 8 s, duty 0.25 (bursts de 2 s) → **ρ_t(burst) = 1.10 > 1 ≥
  ρ_s(burst) = 0.94** (near-margin, fenómeno exhibible).
- **SLA:** L_max = 1.5 ms p99 por evento.
- **B (micro-batch de serving) = 16** para TODOS los métodos.

## 2. Generador de datos (parámetros fijados)

- d = 24 features; ataques = blobs Gaussianos por clase, centros
  equiespaciados en el plano (e1,e2) a distancia ‖μ‖ = attack_radius = 3.0
  (margen de Bayes del gate: error Φ(−3/2) ≈ 6.7%; AUROC objetivo
  teacher ∈ [0.85, 0.97]).
- Severidad ordinal: acumuladas S_k(z) = σ(β·z − c_k) sobre la proyección
  e3, β = 2.0 (pre-registrado; precondición: acc ordinal del teacher >
  marginal 0.5 con margen ≥ 0.10).
- Mezcla α(t) = α0 + γ·(1−α0)·1[burst], α0 = 0.05.
- **Grid γ = {0.0, 0.25, 0.5, 1.0}**; γ=0 es el caso nulo declarado
  (equivalencia LCD vs vanilla esperada por TOST).
- Semillas: train/val/test/rollout = seed, seed+100·split (split offsets
  0/100/200/300), campañas con seed ∈ {0,1,2} (+extensión a {3,4} si el
  poder estadístico lo requiere — decisión ANTES de mirar conclusiones).

## 3. Entrenamiento (θ fijados, sin tuneo sobre métricas de campaña)

- Arquitectura: trunk pre-norm w=128, depth=2, s_dim=3; cabezas teacher
  anchas (head_mult=4); student estrecho 1/8 (w_s=18; corn bottleneck).
- Teacher: epochs=8, batch=512, lr=3e-3 AdamW, wd=1e-2, Kendall
  homoscedástico; temperature scaling por cabeza (grid + sección dorada).
- τ de destilación = 2.0 (idéntico en TODOS los métodos).
- λ_conf = 0.1, α_base = 0.5, α_max = 2.0, α(s) = base + (max−base)·q̂,
  q̂ = clip(q/q_scale, 0, 1), q_scale = 200 ev.
- Curriculum Fase B: época 0 con mezcla global (α const), luego α(s).
- Student (todos los métodos): epochs=4, batch=512, lr=2e-3.
- DAgger (método): 2 iteraciones, oversample×3 de eventos danger
  (cabeza en student concurrency predicción errada), s_vec desplegado.
- G1 (anti-vacuidad): smoothECE ≤ 1% por cabeza × estrato del teacher
  {(calm, warm, hot) × (gate, choice, corn)} en test + PRECONDICIÓN
  ordinal: acc ordinal del corn > marginal + 0.10 (anti-PREDICTOR-CONSTANTE).

## 4. smoothECE y estratos

- Kernel: Gaussiano truncado, radio 5κ, include_self; **κ = 0.05**
  (banda global, pre-registrada en SmoothEceConfig).
- Estratos: cuantiles de q del glass-box train → calm(≤q50) / warm(50–90) /
  hot(>q90); tiles CONGELADOS del split train por celda.
- ECE-transient: ventanas ±2 s alrededor de cada conmutación, n_min=200,
  smoothECE por ventana contra el teacher counterfactual estimado offline.

## 5. Criterios de éxito (go/no-go, pre-registrados)

1. **Validación predictiva (Prop 2):** smoothECE-transient del sistema
   desplegado ≤ RHS de Prop 2 (con δ(s) y Pr[m≤2δ|s] medidos) en ≥90% de
   ventanas (bootstrap jerárquico: ventanas ⊂ trazas ⊂ semillas, B=1000).
   Reporte de tightness: mediana de bound/observed.
2. **LCD (cadena causal):** δ(s) mediana de LCD en buckets de cola
   (warm/hot) < δ(s) de B3a con reducción ≥30% para γ ≥ 0.5; TOST de
   equivalencia en γ=0 (margen ±0.3 puntos smoothECE). Comparaciones
   contra B3a y B3b con Holm (2 primarias, LCD y LCD_dagger).
3. **Router:** a violación de SLA igualada (±1 pt en la frontera Pareto
   interpolada con IC — sweep pre-registrado sobre q̂_on,
   fallback documentado si un método no alcanza el combo), conmutaciones
   del Schmitt ≤ 1/3 de B4-best-q. Ratio vs semi-oracle lookahead:
   diagnóstico (no condición de éxito).
4. **Precondición de capacidad:** C_t, C_s medidas y ρ objetivo
   declarados ANTES de la campaña (§1); ρ no se re-ajusta tras ver
   resultados.

## 6. Multiplicidad y particionado

- Primarias (Holm): (i) LCD vs B3a (hot), (ii) Schmitt·LCD_dagger vs
  B4-best en conmutaciones a viol igualada. Secundarias: tabla completa,
  sin pre-condición de éxito.
- Ablations (3 semillas): α(s) sin curriculum; λ_conf=0; s_dim=0
  (teacher sin S_infra — expectativa: G1 falla, demostración de la
  necesidad del condicionamiento); DAgger sin oversample.

## 7. Entregables de reproducibilidad

- Código + configs + semillas (este repo); microbench κ one-click;
  tablas y figuras one-click (`scripts/make_report.py`); trazas del DES
  guardadas por celda (`results/campaign/cell_g*_s*.json`).

## Apéndice de correcciones

- (ninguna a la fecha del sellado)

### Enmienda 1 (2026-10-21, pilot-driven, ANTES de la parrilla completa) — punto operativo de serving

**Evidencia del pilot (γ=0.5, seed=0):** el punto sellado (service_scale=2.5,
λ_idle=2000, λ_burst=14000, B=16, L_max=1.5 ms) es internamente inconsistente:

1. El servicio de un lote lleno teacher (1237 µs a escala 2.5) consume el 72%
   del presupuesto SLA de 1.5 ms de CABEZA DE LOTE (la primera llegada del
   lote espera la duración completa del forward desplegado). Incluso a carga
   marginal (idle, ρ=0.15) la latencia por evento es ~1.3–1.7 ms.
2. Con ρ_t(burst)=1.083 el overflow hace crecer la cola hasta ~5.3k eventos →
   esperas de cola ~350–450 ms; con ρ_s=0.941 el student TAMBIÉN opera al
   borde → viol% queda degenerado: student_always 90%, teacher 89%, B2/LCD
   ~90% (las diferencias entre métodos se hunden bajo el ruido).
3. El criterio C4 "p99 ≤ 1.5 ms a ≥5k ev/s/núcleo en modo student" es
   inalcanzable en este punto operativo por construcción (no por método).

La calibración F2 (conveyor del microbench) nunca validó el punto operativo
contra el SLA: el smoke usaba service_scale=8 con rates 400/4600 donde el SLA
era trivialmente holgado. El pilot es la validación prevista; falló.

**Enmienda (equivalente en física relativa, discriminatoria en SLA absoluto):**

- `service_scale: 2.5 → 1.0` (física de escala real del microbench).
- Tasas de llegada ×2.5: `λ_idle: 2000 → 5000/s`, `λ_burst: 14000 → 35000/s`.
  Se preservan EXACTO: ρ_t(burst)=1.083, ρ_s(burst)=0.941, duty 0.25,
  periodo 8 s, B=16, L_max=1.5 ms.
- El deadband κ-derivado es INVARIANTE porque κ/Δserv y κ_eventos son ratios
  (κ_eventos = 5.4 en ambas escalas; q_on=0.675, q_off=0.169 idénticos).
- Márgenes por cambio de semilla, arch, horizontes y criterios §5: sin toque.
- Bug de higiene corregido en el pilot: las políticas Schmitt eran ESTADOFUL
  y se reutilizaban entre deploys → cada deploy usa instancia fresca
  (deepcopy). El campo `lat_p50_us` reportaba la media (corregido: p50 real +
  `lat_mean_us` explícito).

Los criterios del §5 se mantienen LITERAL. Este cambio se aplica ANTES de ver
resultados del pilot de calibración-post-enmienda y por partes IGUALES a
todos los métodos (incluye teacher y todos los students).

### Enmienda 2 (2026-10-21) — G1 y smoothECE, umbral de desempeño del teacher

El pilot alcanzó teacher smoothECE gate 1.16–1.47% y corn 1.53–1.76% por
estratos, contra la meta "≤1% por cabeza×estrato". Dos pasos:
1. La implementación del smoothECE cambia a la canónica de B&N (grilla)
   con equivalencia verificada (≤6.6e-5 en el worst case; la DEFINICIÓN no
   cambia). DOCUMENTADO en CONTINUE.md.
2. El umbral de G1 se mantiene en 1% y se añade la re-calibración post-hoc
   de temperaturas MINIMIZANDO el smoothECE de val (objeto del contrato) —
   declarada en `calibration.py::fit_temperatures(objective=...)` con test;
   se adopta solo si el experimento de calibración la confirma en test.
   Si el objetivo ≤1% sigue sin alcanzarse con T-re-fit, se reporta el valor
   real como "precondición de capacidad G1" con brecha documentada — el
   resto de criterios (δ(s), TOST, router) comparan RELATIVO a teacher y no
   dependen de su nivel absoluto.

### Enmienda 3 (2026-10-21) — reloj de servicio del DES determinista

**Evidencia:** con el punto operativo enmendado (E1), el despliegue
student-only seguía violando ~70% del SLA con cola p99 de 78 ms: en vivo, el
student servía ~497 µs/lote ≪ tabla (431 µs) — el DES vivo incluye overheads
de staging (`ascontiguousarray`, `from_numpy`, allocator) que el microbench
amortiza, y la velocidad de máquina depende del estado del box → el punto
operativo ρ NO es reproducible entre corridas/máquina.

**Enmienda:** el reloj de servicio toma la **tabla del microbench**
(muestreo lognormal con la CV medida, seed fija) y el **κ_switch medido se
inserta como costo aparte** por conmutación. El forward real sigue corriendo
para las predicciones del modelo servido, FUERA del reloj (misma convención
del post-proceso). Consistencia: el deadband κ-derivado ya se calcula
contra la tabla (invariante).

Propiedades: (i) reproducibilidad exacta por (seed, tabla, política) —
requisito del paquete repro; (ii) ρ anclado a la tabla: teacher 32.3k ev/s,
student 37.2k ev/s → ρ_t(burst)=1.083 > 1 ≥ ρ_s=0.941 como se selló;
(iii) la filosofía "latencia REAL del forward" se sustituye por
"latencia REAL del benchmark de la arquitectura servida en el hardware
objetivo" — el DES simula el tiempo de serving, la calidad se mide sobre el
forward real desplegado.

Los otros criterios del §5 no cambian.
