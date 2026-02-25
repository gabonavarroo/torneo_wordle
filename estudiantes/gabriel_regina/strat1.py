"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WORDLE STRATEGY — Information Theory Approach                              ║
║  Team: gabriel_regina                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

DISEÑO GENERAL
──────────────
La estrategia tiene dos fases claramente separadas:

  begin_game()  →  Computa el mejor "opener" posible evaluando la entropía
                   ponderada de cada candidato contra el vocabulario COMPLETO.
                   Costo: O(|pool| × |vocab|) ← se paga una sola vez.

  guess()       →  Turno 1: devuelve el opener precomputado.
                   Turno 2+: evalúa el "expected score" para cada candidato
                   en el pool actual y devuelve el que minimiza el costo
                   esperado total.

FUNCIÓN OBJETIVO UNIFICADA
──────────────────────────
Para cada guess candidato g:

    Score(g) = Σ_f  p(f | g) × cost(f)

    donde:
        p(f | g) = masa de probabilidad de los candidatos que producen patrón f
        cost(f)  = 1                        si f = (2,2,...,2)  [ganamos]
                 = 1 + f̂(|candidatos_f|)   en otro caso

    f̂(N) ≈ guesses adicionales esperados cuando quedan N candidatos.

Esta función hace la transición exploración→explotación de forma automática:
  - Con muchos candidatos: p_win pequeño, domina minimizar f̂ ≈ maximizar entropía
  - Con pocos candidatos: p_win grande puede justificar adivinar directamente

MODOS
─────
  uniform:   todos los candidatos pesan 1/N al calcular p(f|g)
  frequency: cada candidato pesa config.probabilities[w], renormalizado

NO-PALABRAS
───────────
Permitidas por el torneo. Se usan en el opener y en turno 2 cuando quedan
muchos candidatos. Nunca pueden ganar directamente, pero pueden dar más
información que cualquier palabra real al cubrir letras más frecuentes
sin restricciones fonológicas del español.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

from strategy import Strategy, GameConfig
from wordle_env import feedback, filter_candidates


# ══════════════════════════════════════════════════════════════════════════════
# Constantes de rendimiento
# Ajustar aquí si hay problemas de timeout en vocabularios muy grandes.
# ══════════════════════════════════════════════════════════════════════════════

# Opener: cuántas palabras del vocabulario evaluar como candidatos (ordenadas
# por número de letras únicas — más letras únicas = más informativa).
MAX_OPENER_VOCAB_POOL = 200

# Opener: cuántas no-palabras generar y evaluar.
MAX_NON_WORDS = 80

# guess() turno 2+: máximo de guesses en el pool de evaluación.
MAX_GUESS_POOL = 250

# guess() turno 2: cuántas no-palabras añadir al pool cuando quedan muchos
# candidatos y vale la pena explorar fuera del vocabulario.
MAX_NON_WORDS_TURN2 = 50

# Umbral: si quedan más candidatos que esto en turno 2, añadimos no-palabras.
NON_WORD_THRESHOLD_TURN2 = 30


# ══════════════════════════════════════════════════════════════════════════════
# Estimador de guesses futuros  f̂(N)
# ══════════════════════════════════════════════════════════════════════════════

def _f_hat(n: int) -> float:
    """
    Estima cuántos guesses adicionales se necesitan en promedio cuando quedan
    N candidatos, asumiendo juego óptimo de ahí en adelante.

    Derivación intuitiva:
      - N=1 → ya sabes la respuesta, 1 guess basta.
      - N=2 → 50% de ganar en el siguiente, promedio = 1.5 guesses.
      - N=3 → ~2 guesses esperados.
      - N>3 → crece como log₂(N) con constante empírica calibrada para
              vocabularios de español de ~1k-6k palabras.

    Esta función actúa como V(N) en la ecuación de Bellman:
        Score(g) = Σ_f p(f|g) × (1 + f̂(|C_f|))
    """
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 2.0
    # Calibración empírica. Con strategy óptima en inglés:
    #   ~13k palabras → ~3.5 guesses → f̂ necesita mapear log₂(13k)≈13.6 bits a 3.5
    # Para español (~4k palabras), valores similares funcionan bien.
    return max(1.0, math.log2(n) * 0.5 + 0.8)


# ══════════════════════════════════════════════════════════════════════════════
# Clase principal
# ══════════════════════════════════════════════════════════════════════════════

class InformationStrategy(Strategy):

    @property
    def name(self) -> str:
        return "InformationStrategy_gabriel_regina"

    # ──────────────────────────────────────────────────────────────────────────
    # begin_game — inicialización y cómputo del opener
    # ──────────────────────────────────────────────────────────────────────────

    def begin_game(self, config: GameConfig) -> None:
        """
        Se llama una vez por juego. Presupuesto: la mayor parte de los 5s.

        Pasos:
          1. Guardar configuración y vocabulario.
          2. Calcular pesos normalizados del vocabulario completo.
          3. Generar pool de no-palabras (reutilizable en turn 2).
          4. Encontrar el mejor opener evaluando entropía contra vocab completo.
        """
        self._config   = config
        self._vocab    = list(config.vocabulary)
        self._vocab_set = set(config.vocabulary)
        self._probs    = config.probabilities
        self._wl       = config.word_length
        self._mode     = config.mode

        # Pesos del vocabulario completo, normalizados a suma=1.
        # En frequency: usa config.probabilities.
        # En uniform: todos iguales.
        self._vocab_weights = self._normalize_weights(self._vocab)

        # No-palabras precomputadas (se reutilizan en guess() si es necesario).
        self._non_words = self._generate_non_words()

        # El opener óptimo — cómputo principal de begin_game.
        self._opener = self._compute_opener()

    # ──────────────────────────────────────────────────────────────────────────
    # Pesos y normalización
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_weights(self, candidates: list[str]) -> dict[str, float]:
        """
        Calcula pesos normalizados para un subconjunto de candidatos.

        La renormalización es ESENCIAL cuando trabajamos con un subconjunto
        del vocabulario (ej. candidatos restantes tras filtrar). Sin ella,
        los pesos no suman 1 y las probabilidades p(f|g) son inválidas.

        frequency: peso base = config.probabilities[w]  (ya incorpora sigmoide
                               del corpus OpenSLR, aplicada en lexicon.py)
        uniform:   peso base = 1.0  (todos iguales → se normaliza a 1/N)
        """
        if self._mode == "frequency":
            raw = {w: self._probs.get(w, 1e-10) for w in candidates}
        else:
            raw = {w: 1.0 for w in candidates}

        total = sum(raw.values())
        if total == 0:
            n = len(candidates)
            return {w: 1.0 / n for w in candidates}

        return {w: v / total for w, v in raw.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Entropía ponderada  H(g | candidatos, pesos)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_entropy(
        self,
        guess: str,
        candidates: list[str],
        weights: dict[str, float],
    ) -> float:
        """
        Entropía de Shannon ponderada de la partición producida por `guess`.

        Para cada patrón de feedback posible f:
            p(f | guess) = Σ_{w ∈ candidatos : feedback(w, guess) = f} weights[w]

        H(guess) = -Σ_f  p(f) · log₂(p(f))

        Interpretación:
          H alta → guess produce muchas particiones de tamaño similar → más información.
          H baja → una partición domina → poco aprendizaje en promedio.

        En modo uniform: reduce a entropía uniforme clásica (cuenta palabras por grupo).
        En modo frequency: las palabras frecuentes pesan más → el opener óptimo
                           prioriza separar bien las palabras comunes entre sí.

        NOTA: Esta función se usa solo para el opener (begin_game).
              En guess() usamos _expected_score() que es más general.
        """
        partition: dict[tuple, float] = defaultdict(float)

        for w in candidates:
            pat = feedback(w, guess)
            partition[pat] += weights.get(w, 0.0)

        total = sum(partition.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for mass in partition.values():
            p = mass / total
            if p > 0.0:
                entropy -= p * math.log2(p)

        return entropy

    # ──────────────────────────────────────────────────────────────────────────
    # Generación de no-palabras
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_non_words(self) -> list[str]:
        """
        Genera no-palabras informativas combinando las letras más frecuentes
        del vocabulario, sin restricciones fonológicas.

        Ventaja vs palabras reales:
          Las palabras reales del español tienen restricciones (no puedes tener
          ciertas combinaciones). Una no-palabra puede cubrir las wl letras más
          frecuentes exactamente, maximizando la información extraíble.

        Costo: nunca puede ganar directamente (prob_win = 0).
          → Solo vale usarlas cuando p_win es pequeño de todas formas
            (openers, turno 2 con muchos candidatos).

        Algoritmo:
          1. Calcular frecuencia ponderada de cada letra por posición.
          2. Seleccionar top_letters globales (letras más frecuentes en total).
          3. Para cada combinación C(top, wl) de letras sin repetición:
             - Asignar cada letra a la posición donde tiene mayor frecuencia
               (asignación greedy, maximiza la señal posicional).
          4. Devolver hasta MAX_NON_WORDS no-palabras distintas.
        """
        wl   = self._wl
        vocab = self._vocab
        weights = self._vocab_weights

        # ── Paso 1: frecuencia ponderada letra×posición ──────────────────────
        pos_freq: list[dict[str, float]] = [defaultdict(float) for _ in range(wl)]
        for w in vocab:
            w_weight = weights.get(w, 1.0 / max(len(vocab), 1))
            for i, ch in enumerate(w):
                pos_freq[i][ch] += w_weight

        # ── Paso 2: frecuencia global (suma sobre posiciones) ────────────────
        overall: dict[str, float] = defaultdict(float)
        for i in range(wl):
            for ch, freq in pos_freq[i].items():
                overall[ch] += freq

        top_letters = [ch for ch, _ in sorted(overall.items(), key=lambda x: -x[1])]

        # ── Paso 3: combinaciones sin repetición de las top letras ───────────
        # Tomamos wl + 5 letras para dar variedad sin explotar combinatoriamente.
        # C(10, 5) = 252   C(11, 6) = 462   → manejable
        pool_size = min(wl + 5, len(top_letters))
        candidate_letters = top_letters[:pool_size]

        non_words: list[str] = []
        seen: set[str] = set()

        for combo in itertools.combinations(candidate_letters, wl):
            # Asignación greedy: para cada posición, la letra disponible con
            # mayor frecuencia posicional.
            remaining = list(combo)
            assignment = [''] * wl

            for i in range(wl):
                best_ch = max(remaining, key=lambda ch: pos_freq[i].get(ch, 0.0))
                assignment[i] = best_ch
                remaining.remove(best_ch)

            nw = "".join(assignment)

            # Solo añadir si es nueva y no está en el vocabulario real.
            if nw not in seen and nw not in self._vocab_set:
                seen.add(nw)
                non_words.append(nw)

            if len(non_words) >= MAX_NON_WORDS:
                break

        return non_words

    # ──────────────────────────────────────────────────────────────────────────
    # Cómputo del opener
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_opener(self) -> str:
        """
        Encuentra el guess inicial óptimo maximizando H(g) contra el vocabulario
        COMPLETO con sus pesos.

        IMPORTANTE: El vocabulario del lado "secreto" NUNCA se muestrea.
        Muestrearlo distorsionaría la entropía real. Solo el pool de openers
        a evaluar puede limitarse sin perder mucho (el top opener y el top
        del sample suelen coincidir).

        Pool de evaluación:
          - Palabras del vocabulario con más letras únicas (más letras distintas
            = más hipótesis testeadas por guess → mayor entropía esperada).
          - No-palabras generadas en _generate_non_words().

        Complejidad: O(|pool| × |vocab|) llamadas a feedback().
        Con pool=280, vocab=4000: ~1.1M llamadas → ~1-2 segundos en Python.
        """
        vocab   = self._vocab
        weights = self._vocab_weights  # pesos sobre vocab completo

        # ── Pool de palabras del vocabulario ─────────────────────────────────
        # Criterio: letras únicas (descendente). Una palabra con 5 letras
        # distintas testea 5 hipótesis; una con 3 únicas + 2 repetidas, solo 3.
        vocab_pool = sorted(vocab, key=lambda w: len(set(w)), reverse=True)
        vocab_pool = vocab_pool[:MAX_OPENER_VOCAB_POOL]

        # ── Pool completo: palabras + no-palabras ─────────────────────────────
        opener_pool = vocab_pool + self._non_words

        # ── Evaluación: H(g) contra vocab completo ───────────────────────────
        best_guess   = vocab[0]
        best_entropy = -1.0

        for g in opener_pool:
            h = self._compute_entropy(g, vocab, weights)
            if h > best_entropy:
                best_entropy = h
                best_guess   = g

        return best_guess

    # ──────────────────────────────────────────────────────────────────────────
    # Expected score  Score(g | candidatos, pesos)
    # ──────────────────────────────────────────────────────────────────────────

    def _expected_score(
        self,
        guess: str,
        candidates: list[str],
        weights: dict[str, float],
    ) -> float:
        """
        Costo esperado total (en guesses) de usar `guess` en el estado actual.

        Ecuación de Bellman aproximada:
            Score(g) = Σ_f  p(f | g) × cost(f)

            cost(f = all-green) = 1                ← ganamos en este turno
            cost(f ≠ all-green) = 1 + f̂(|C_f|)   ← 1 guess + los que faltan

        Donde:
            p(f | g) = Σ_{w ∈ C_f} weights[w]     masa de prob. del patrón f
            C_f      = candidatos que producen patrón f dado guess g
            f̂(N)     = estimación de guesses futuros para N candidatos

        Propiedades deseables:
          ✓ Cuando p_win es alta (pocos candidatos, uno muy probable):
            el término de ganar domina → estrategia explota.
          ✓ Cuando p_win es baja (muchos candidatos):
            domina minimizar f̂(|C_f|) ≈ maximizar entropía → explora.
          ✓ La transición es AUTOMÁTICA, sin switch manual.
          ✓ Las no-palabras tienen p_win=0 pero pueden minimizar Σ f̂(|C_f|).
          ✓ Funciona igual en uniform y frequency (solo cambian los pesos).

        Minimizar Score(g) es el objetivo.
        """
        win_pat = tuple([2] * self._wl)

        # Particionar candidatos por patrón de feedback
        partition: dict[tuple, list[str]] = defaultdict(list)
        for w in candidates:
            pat = feedback(w, guess)
            partition[pat].append(w)

        total_weight = sum(weights.get(w, 0.0) for w in candidates)
        if total_weight == 0.0:
            total_weight = 1.0

        score = 0.0
        for pat, group in partition.items():
            # Masa de probabilidad de este patrón
            group_weight = sum(weights.get(w, 0.0) for w in group)
            p_f = group_weight / total_weight

            if pat == win_pat:
                # Este patrón significa victoria — costo = 1 guess
                # (p_f aquí es exactamente p_win si guess está en candidatos)
                score += p_f * 1.0
            else:
                # Gastamos 1 guess y nos quedan len(group) candidatos
                score += p_f * (1.0 + _f_hat(len(group)))

        return score

    # ──────────────────────────────────────────────────────────────────────────
    # guess — lógica principal de decisión
    # ──────────────────────────────────────────────────────────────────────────

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        """
        Selecciona el siguiente guess dado el historial de (guess, feedback).

        Flujo de decisión:
          Turno 1 (sin historial):
            → Devuelve el opener precomputado en begin_game(). Instantáneo.

          Turno 2+ (con historial):
            1. Filtra candidatos consistentes con todo el feedback acumulado.
            2. Casos triviales: 1 candidato → devuélvelo;
                                2 candidatos → devuelve el más probable.
            3. Construye pool de evaluación:
               - Siempre incluye todos los candidatos actuales (pueden ganar).
               - En turno 2 con ≥30 candidatos: añade no-palabras precomputadas.
            4. Evalúa expected_score(g) para cada g en el pool.
            5. Devuelve el g con menor score.
        """

        # ── Turno 1: opener precomputado ──────────────────────────────────────
        if not history:
            return self._opener

        # ── Filtrar candidatos ─────────────────────────────────────────────────
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            # No debería ocurrir con feedback consistente, pero por seguridad:
            return self._vocab[0]

        # ── Casos triviales ────────────────────────────────────────────────────
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) == 2:
            # Con 2 candidatos siempre adivinamos el más probable.
            # En uniform son iguales → elige el primero alfabéticamente.
            return max(candidates, key=lambda w: self._probs.get(w, 0.0))

        # ── Pesos renormalizados sobre los candidatos actuales ─────────────────
        weights = self._normalize_weights(candidates)

        # ── Construir pool de evaluación ───────────────────────────────────────
        turn = len(history) + 1

        # Base: todos los candidatos actuales. SIEMPRE se incluyen porque pueden
        # ganar directamente (tienen p_win > 0).
        guess_pool = list(candidates)

        # Turno 2 con muchos candidatos: las no-palabras pueden dar más información
        # que cualquier palabra real al no tener restricciones fonológicas.
        # A partir del turno 3, la ganancia de las no-palabras se vuelve marginal
        # y el costo de evaluar más candidatos no vale la pena.
        if turn == 2 and len(candidates) >= NON_WORD_THRESHOLD_TURN2:
            guess_pool = list(candidates) + self._non_words[:MAX_NON_WORDS_TURN2]

        # Cap de pool para mantener velocidad en casos con muchos candidatos.
        # Priorizar siempre los candidatos reales sobre las no-palabras.
        if len(guess_pool) > MAX_GUESS_POOL:
            cands_set = set(candidates)
            non_cands = [g for g in guess_pool if g not in cands_set]
            # Tomar todos los candidatos + no-palabras hasta completar el cap
            guess_pool = candidates + non_cands[:max(0, MAX_GUESS_POOL - len(candidates))]

        # ── Evaluar expected score ─────────────────────────────────────────────
        best_guess = candidates[0]
        best_score = float('inf')

        for g in guess_pool:
            score = self._expected_score(g, candidates, weights)
            if score < best_score:
                best_score = score
                best_guess = g

        return best_guess