# Plan maestro: estrategia híbrida para mínimo promedio de guesses (6 modalidades)

## Objetivo competitivo
Minimizar el promedio global de intentos en:
- 4 letras (uniform, frequency)
- 5 letras (uniform, frequency)
- 6 letras (uniform, frequency)

Bajo límite online de ~5s/juego, moviendo casi todo el costo a precomputación offline.

---

## 1) Qué ya está bien en la base actual (y debe conservarse)

### A. Árboles precomputados por turnos tempranos
El `strategy.py` actual ya usa tablas T2/T3 (y T4 en 4 letras), lo cual es correcto: lookup O(1) en runtime.

### B. Uso de no-palabras/probes
Tanto en `parallel_dp.py` como en `strategy.py` se contempla el valor de guesses fuera del set candidato para romper clusters.

### C. Fallback dinámico para estados fuera de tabla
Es valioso tener fallback para estados no cubiertos por la precomputación.

---

## 2) Gap principal de la implementación actual

### Gap #1: criterio aproximado en runtime
La estrategia actual usa `_expected_score` con `_f_hat(n)` (aprox. de costo futuro). Esto puede ser subóptimo en ramas difíciles.

### Gap #2: cobertura desigual del árbol
Hay T4 completo solo en 4 letras. En 5/6 letras puede caer mucho en fallback.

### Gap #3: pool recortado por slicing fijo
`self._vocab[:400]` introduce sesgo y puede perder probes críticos para separar clusters.

### Gap #4: probes poco contextualizados por rama
La generación de probes del runtime es genérica; falta branch-specific usando restricciones finas acumuladas.

---

## 3) Estrategia ideal híbrida (versión competitiva)

## Fase offline (sin límite práctico de tiempo)

### 3.1 Openers óptimos por modalidad
- Mantener búsqueda exhaustiva por modalidad (`wl`, `mode`), optimizando esperanza de pasos.
- En `uniform`: prioridad media + robustez worst-case.
- En `frequency`: esperanza ponderada por probabilidad real.

### 3.2 Árbol principal por estados (DP exacto)
- Construir árbol/tabla por estado con:
  - `best_guess`
  - `expected_remaining`
  - `max_bucket` (riesgo de cluster)
- Reutilizar memo por estado canónico (set de candidatos + restricciones), no por camino textual.

### 3.3 Biblioteca de probes por firma de estado
Precomputar probes por clases de estado:
- `high-entropy-global` (muchas letras inciertas)
- `position-disambiguation` (amarillas conflictivas)
- `cluster-breaker` (candidatos parecidos)
- `endgame-2turn-safe` (maximizar singletons y minimizar peor caso)

### 3.4 Cobertura objetivo por formato
- 4 letras: cobertura casi total hasta T5.
- 5 letras: cobertura fuerte hasta T4/T5 en ramas grandes.
- 6 letras: cobertura fuerte T1–T4 para ramas de alta masa de probabilidad.

---

## Fase online (torneo, 5s por juego)

## 4) Política por turno y por tipo de estado

### T1 (siempre tabla)
- Usar opener óptimo precomputado por `wl` y `mode`.

### T2 (tabla exacta por feedback del opener)
- Lookup O(1) del mejor guess, permitiendo palabra o no-palabra.
- En ramas de alta incertidumbre, preferir máxima entropía efectiva (partición más uniforme).

### T3 (árbol híbrido)
Seleccionar según tipo de estado:
1. **Si estado existe en tabla**: usar `best_guess` precomputado.
2. **Si no existe**:
   - Si `n_candidates` alto: score híbrido entropía + minimax bucket.
   - Si `n_candidates` medio y `frequency`: añadir componente probabilidad posterior.
   - Si cluster fuerte: usar `cluster-breaker probe` aunque no sea candidata.

### T4
- Regla principal: resolver, no solo informar.
- Si `P(top_candidate)` alta y riesgo de cluster bajo: guess de máxima probabilidad.
- Si riesgo alto y quedan pocos turnos: probe minimax que garantice separación en T5.

### T5/T6 endgame (crítico)
- Si `guesses_left >= n_candidates`: jugar candidata más probable.
- Si `n_candidates > guesses_left`:
  - elegir sonda que minimice `max_bucket` y maximice singletons.
  - luego jugar respuesta más probable consistente.
- En `frequency`, desempatar por masa de probabilidad eliminada esperada.

---

## 5) Función de decisión híbrida recomendada

Para estados sin tabla exacta, usar:

`score(g) = a_t * E[depth|g] + b_t * max_bucket(g) - c_t * P(win_next|g) - d_t * H_partition(g)`

con pesos por turno `t`:
- Early (T2/T3): subir `d_t` y bajar `c_t`.
- Mid (T3/T4): balance `a_t`, `b_t`, `d_t`.
- Late (T4+): subir fuerte `b_t` y `c_t`.

Esto evita caer en entropía pura cuando ya conviene cerrar.

---

## 6) Diferenciación por modalidad

### Uniform
- Optimizar media + robustez worst-case.
- Dar más peso a `max_bucket`.

### Frequency
- Optimizar esperanza ponderada.
- Subir peso de `P(win_next)` y masa posterior.

---

## 7) Cambios concretos sugeridos al código actual

1. **Unificar esquema de tablas**
   - Exportar/consumir JSON de estado canónico homogéneo para T2–T5 (idealmente más profundo en 5/6).

2. **Eliminar pool fijo `[:400]` en fallback**
   - Sustituir por pool adaptativo por riesgo de estado.

3. **Reemplazar `_gen_probe_nonwords` por selector de probes precomputados**
   - Indexado por firma de restricciones.

4. **Ampliar endgame**
   - No solo cuando `guesses_left <= 2`; activar política de minimax desde T4 en clusters.

5. **Precompute orientado a cobertura de masa**
   - En 6 letras, priorizar ramas con mayor probabilidad acumulada para maximizar impacto en promedio.

---

## 8) Roadmap de implementación (recomendado)

### Etapa 1 (rápida, alto impacto)
- Mantener tablas actuales.
- Cambiar fallback a score híbrido + pool adaptativo + minimax temprano.

### Etapa 2
- Generar tabla T4/T5 para 5 letras y T4 robusto para 6 letras.
- Integrar biblioteca de probes por estado.

### Etapa 3 (versión top)
- Rehacer pipeline con DP exacto por estado canónico y export masivo comprimido.
- Runtime casi puro lookup con fallback mínimo.

---

## 9) Métricas para decidir si realmente “ya es la mejor”

Medir por modalidad:
- mean guesses
- solve rate <= 6
- percentil 90 de guesses
- tasa de fallback (qué tanto no entró a tabla)
- tiempo promedio por juego

Criterio de éxito competitivo:
- mejor promedio global en Borda esperado y menor varianza entre modalidades.

---

## Conclusión
La mejor estrategia competitiva aquí no es “entropía pura” ni “probabilidad pura”, sino:
1) lookup masivo precomputado en early/mid,
2) política híbrida adaptativa por estado,
3) probes no-palabra específicos para romper clusters,
4) endgame minimax con sensibilidad a probabilidad en `frequency`.

Con ese diseño, se minimiza el promedio y también se blindan los casos difíciles que suelen decidir torneos.
