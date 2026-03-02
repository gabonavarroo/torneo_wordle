# Estrategia gabriel_regina (RG series)

Este README resume todo el diseño, precomputación y lógica por turno de la estrategia final (`strategy-final.py`, clase `RG2_gabriel_regina`), así como las decisiones que se probaron en iteraciones previas (RG3–RG5). El objetivo fue optimizar el promedio de intentos y la tasa de resolución bajo las reglas del torneo oficial: 6 rondas {4,5,6} x {uniform,frequency}, 5 s por juego, 1 CPU, distribución perturbada (`shock=0.05`) en frequency.

## Precomputación (offline)

- **Openers**: búsqueda exhaustiva (`fast_opener_search.py`) sobre 27^wl para máxima entropía (se permiten no‑palabras). Resultado usado:
  - 4L: `aore`
  - 5L: `sareo`
  - 6L: `ceriao` (frequency se usó temporalmente `carieo`; tablas actuales embebidas con opener óptimo `ceriao`)
- **T2 óptimo por branch** (`parallel_t2.py`):
  - 4L: 56 branches (patrón T1) → búsqueda exacta en 27^4 contra el subconjunto; incluye no‑palabras.
  - 5L: 172 branches → prefiltrado entropía vectorizada, luego exacto sobre top‑pool.
  - 6L: 409 branches → vocab + no‑palabras evidentes (exhaustivo 27^6 es inviable).
  - Salida: `t2_table_{4,5,6}_{uniform,frequency}.json`.
- **T3 óptimo por estado** (`parallel_t3_v5.py`):
  - Estados = (pat_T1, pat_T2); clasificados como trivial/two/cluster/few/many.
  - Para n≤12 candidatos: expected cost exacto (no entropía proxy). Para clusters (≤2 posiciones variables): buster óptimo.
  - Salida: `t3_table_{4,5,6}_{uniform,frequency}.json`.
- Estas tablas están embebidas en `strategy-final.py` (_EMBEDDED_T2/_EMBEDDED_T3) y fueron verificadas byte‑a‑byte contra los JSON originales (sin diferencias).

## Arquitectura por turno (runtime)

- **T1**: opener fijo por (wl, mode).
- **T2**: lookup O(1) en tabla; fallback:
  - Frequency: búsqueda dinámica por expected cost (vocab sample + non‑words).
  - Uniform: entropía sobre vocab (pool cap).
- **T3**: lookup O(1) en tabla; fallback:
  - n≤1 → trivial; n=2 → más probable.
  - Frequency: si `p_best > 0.60`, adivinar directo; si no, expected cost con probes.
  - Uniform: entropía.
  - En RG5 se añadió un buster runtime para clusters pequeños n≤8, pero la versión final RG2 no lo usa (confía en la tabla).
- **T4**:
  - n≤2 → directo.
  - n≤10 → buscar safe guess (max group size=2). En frequency se compara `E[direct]` vs `E[safe]`.
  - Si no hay safe: frequency directo si `p_best > 0.60`, si no expected cost; uniform entropía.
- **T5**:
  - n≤2 → directo.
  - n=3 → safe guess (grupo≤1), si no directo.
  - n≥4 → frequency expected cost; uniform entropía.
- **T6+**: siempre el más probable (regla óptima).

## Decisiones clave

- **Precompute hasta T3**: cubre el 80% del valor con bajo costo; T4+ se resuelve mejor en runtime adaptando a turns_left y n_cands.
- **No‑palabras permitidas**: abren particiones más balanceadas en T1/T2 y probes T3/T4 (frequency).
- **Safe guess**: garantiza P(fallo)=0 en T4/T5 cuando existe; prioridad sobre entropía si se encuentra.
- **Umbrales frequency**: T3 directo 0.60, T4 directo 0.60 para aprovechar p_best alta aun con shock.
- **Shock robustness**: normalizar probs en el set activo y preferir probes sobre entropía en frequency reduce sensibilidad a perturbaciones.

## Iteraciones (RG2 → RG5)

- **RG2 (final base)**: tablas T2/T3 externas; frequency fallback con probes dinámicos; uniform fallback entropía; umbrales T3/T4 0.65/0.60 (ajustados luego a 0.60/0.60); sin cluster runtime.
- **RG3**: aplicó probes también en uniform; redujo umbral T3 freq. Ganó algunos early hits pero aumentó varianza en uniform.
- **RG4**: revirtió uniform a entropía (estabilidad) y mantuvo probes en freq.
- **RG5**: añadió cluster-buster runtime n≤8 en T3 fallback. No se activaba casi nunca porque las tablas ya cubrían casi todos los estados.
- **strategy-final.py**: embebe tablas y usa la lógica RG2 afinada (probes solo en freq, entropía en uniform, umbral freq T3 0.60).

## Ahorro de cómputo y límites

- Tablas cargadas una vez por proceso; sin I/O en juegos posteriores.
- Pooles capados (300–500 palabras) y probes limitados (≤20) para mantener <5 s/juego.
- Safe guess busca primero en candidatos ordenados por prob, luego en vocab; corta en cuanto encuentra.
- No se precomputó T4+ porque los conjuntos son pequeños y el runtime exacto es suficientemente rápido.

## Validación

- Comparación de tablas embebidas vs JSON originales: 0 diferencias (t2/t3 para todas las longitudes y modos).
- Torneos con `--official --shock 0.05 --num-games 100` muestran medias ~3.7 y solve% ≈99%+ con RG2/RG4/RG5; RG2/strategy-final es la versión entregable estable.

## Cómo correr

```bash
# Torneo oficial rápido (100 juegos por ronda)
python3 tournament.py --official --shock 0.05 --num-games 100 --team gabriel_regina
```

La estrategia a entregar debe estar en `estudiantes/gabriel_regina/strategy.py` con la clase final; `strategy-final.py` contiene los embeds y la lógica descrita arriba.***
