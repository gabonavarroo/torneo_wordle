""" Version 1.02
╔══════════════════════════════════════════════════════════════════════════════╗
║  WORDLE STRATEGY — Information Theory Approach  [v2 — timeout fixed]        ║
║  Team: gabriel_regina                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

DIAGNÓSTICO Y FIX DE TIMEOUT
──────────────────────────────
El profiling reveló que begin_game tardaba:
  4 letras (1,853 palabras):  ~1.35s  ✓
  5 letras (4,546 palabras):  ~3.74s  ⚠️  al límite
  6 letras (6,016 palabras):  ~5.60s  ✗  timeout antes del primer guess

Causa: O(|pool_openers| × |vocab|) llamadas a feedback() en cada begin_game.
       280 openers × 6,016 palabras = 1.68M llamadas → 5.6s en Python puro.

Dos fixes aplicados juntos:

  FIX 1 — CACHÉ DE CLASE
  ───────────────────────
  El opener es DETERMINISTA dado (vocabulario, modo). En un torneo, todos los
  juegos de la misma ronda reciben exactamente el mismo vocabulario. Sin caché,
  cada juego paga ~5.6s de begin_game. Con caché de clase, solo el PRIMER juego
  de cada ronda paga el costo; los siguientes recuperan el resultado en O(1).

  Clave del caché: (hash(vocab_tuple), mode_str)
  Valor:           (opener_str, non_words_list)

  FIX 2 — POOL REDUCIDO
  ──────────────────────
  De ~280 openers evaluados → ~100 (70 palabras + 30 no-palabras).
  Impacto en calidad: mínimo. El opener óptimo real casi siempre está entre
  los top-70 por número de letras únicas. La entropía es una función suave:
  el top-5 de openers tiene valores muy similares.

  Tiempos esperados tras el fix (PRIMER juego de cada ronda):
    4 letras: ~0.6s   (100 × 1,853 = 185k llamadas)
    5 letras: ~1.5s   (100 × 4,546 = 455k llamadas)
    6 letras: ~2.0s   (100 × 6,016 = 602k llamadas)
  Juegos siguientes: ~0s (caché hit)

FUNCIÓN OBJETIVO — Expected Score
───────────────────────────────────
Para cada guess candidato g:

    Score(g) = Σ_f  p(f|g) × cost(f)

    cost(all-green) = 1                  ← ganamos en este turno
    cost(otro f)    = 1 + f̂(|C_f|)      ← 1 guess gastado + los que faltan

    p(f|g) = Σ_{w ∈ C_f} weights[w]     (masa de probabilidad del patrón f)
    f̂(N)   = guesses adicionales esperados con N candidatos

Transición exploración→explotación automática:
  - Muchos candidatos: p_win pequeño, domina minimizar f̂ ≈ maximizar entropía
  - Pocos candidatos:  p_win significativo, puede valer adivinar directamente
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

from strategy import Strategy, GameConfig
from wordle_env import feedback, filter_candidates


# ══════════════════════════════════════════════════════════════════════════════
# Constantes de rendimiento
# Calibradas para mantener begin_game (primer juego) < 2.5s en vocab de 6k.
# ══════════════════════════════════════════════════════════════════════════════

# Opener: palabras del vocabulario a evaluar (top por letras únicas).
# 70 × 6,016 = 421k llamadas → ~1.4s. Deja margen para no-palabras.
MAX_OPENER_VOCAB_POOL = 70

# Opener: no-palabras a evaluar. 30 × 6,016 = 180k llamadas adicionales → ~0.6s.
# Total primer juego 6 letras: ~2.0s. ✓
MAX_NON_WORDS = 30

# guess() turno 2+: máximo de guesses en el pool de evaluación.
# Con 250 candidatos × 250 pool: 62k llamadas → ~0.2s. Bien dentro del límite.
MAX_GUESS_POOL = 250

# Turno 2: no-palabras extra a añadir cuando quedan muchos candidatos.
MAX_NON_WORDS_TURN2 = 40

# Umbral: añadir no-palabras en turno 2 solo si quedan ≥ N candidatos.
# Con pocos candidatos la ganancia de no-palabras es marginal.
NON_WORD_THRESHOLD_TURN2 = 30


# ══════════════════════════════════════════════════════════════════════════════
# Estimador de guesses futuros  f̂(N)
# ══════════════════════════════════════════════════════════════════════════════

def _f_hat(n: int) -> float:
    """
    Estima cuántos guesses adicionales se necesitan en promedio con N candidatos.

    Calibración empírica para vocabularios de español (~1k-6k palabras):
      N=1 → 1.0   (ya sabes la respuesta)
      N=2 → 1.5   (50% de acertar en el siguiente)
      N=3 → 2.0   (~2 guesses esperados)
      N>3 → log₂(N)×0.5 + 0.8  (crece suavemente)

    Esta función es V(N) en la aproximación de la ecuación de Bellman:
        Score(g) = Σ_f p(f|g) × (1 + f̂(|C_f|))
    """
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 2.0
    return max(1.0, math.log2(n) * 0.5 + 0.8)


# ══════════════════════════════════════════════════════════════════════════════
# Clase principal
# ══════════════════════════════════════════════════════════════════════════════

class InformationStrategy(Strategy):
    """
    Estrategia basada en teoría de la información con expected score.

    El atributo de clase _opener_cache persiste entre instancias (entre juegos
    de la misma ronda del torneo), eliminando el costo de recomputar el opener.
    """

    # ── Caché compartido entre todas las instancias ───────────────────────────
    # Clave:  (hash(vocab_tuple), mode_str)
    # Valor:  (opener_str, non_words_list)
    #
    # Por qué a nivel de clase:
    # El torneo crea una nueva instancia de la estrategia por juego.
    # Sin caché de clase, cada instancia recomputaría el opener desde cero.
    # Con caché de clase, el cómputo ocurre una sola vez por ronda.
    _opener_cache: dict[tuple, tuple[str, list[str]]] = {}

    @property
    def name(self) -> str:
        return "InformationStrategy_gabriel_regina"

    # ──────────────────────────────────────────────────────────────────────────
    # begin_game
    # ──────────────────────────────────────────────────────────────────────────

    def begin_game(self, config: GameConfig) -> None:
        """
        Inicialización por juego.

        Con caché: O(1) para juegos 2, 3, 4... de la misma ronda.
        Sin caché (primer juego): O(pool × vocab) — el costo principal.

        Clave del caché: (hash(config.vocabulary), config.mode)
          - config.vocabulary es una tuple → hasheable directamente.
          - Incluir mode porque uniform y frequency pueden dar openers distintos.

        Nota sobre distribution shock en frequency mode:
          Las probabilidades varían ≤5% entre juegos, pero el opener óptimo
          es robusto a perturbaciones pequeñas. Reutilizar el opener cacheado
          es una aproximación válida que ahorra el recómputo completo.
        """
        self._config    = config
        self._vocab     = list(config.vocabulary)
        self._vocab_set = set(config.vocabulary)
        self._probs     = config.probabilities
        self._wl        = config.word_length
        self._mode      = config.mode

        # Intentar recuperar del caché
        cache_key = (hash(config.vocabulary), config.mode)

        if cache_key in InformationStrategy._opener_cache:
            self._opener, self._non_words = InformationStrategy._opener_cache[cache_key]
            return

        # Primer juego de esta configuración: calcular todo
        self._vocab_weights = self._normalize_weights(self._vocab)
        self._non_words     = self._generate_non_words()
        self._opener        = self._compute_opener()

        # Guardar en caché
        InformationStrategy._opener_cache[cache_key] = (
            self._opener,
            self._non_words,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Pesos y normalización
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_weights(self, candidates: list[str]) -> dict[str, float]:
        """
        Pesos normalizados para un subconjunto de candidatos.

        La renormalización es esencial al trabajar con subconjuntos del
        vocabulario (candidatos restantes tras filtrar). Sin ella, las
        probabilidades p(f|g) no suman 1 y el expected score es incorrecto.

        frequency: usa config.probabilities[w], que ya incorpora la
                   transformación sigmoide sobre frecuencias de corpus (lexicon.py).
        uniform:   todos iguales → equivale a 1/N tras normalizar.
        """
        if self._mode == "frequency":
            raw = {w: self._probs.get(w, 1e-10) for w in candidates}
        else:
            raw = {w: 1.0 for w in candidates}

        total = sum(raw.values())
        if total == 0:
            n = max(len(candidates), 1)
            return {w: 1.0 / n for w in candidates}

        return {w: v / total for w, v in raw.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Entropía ponderada  (solo para el opener)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_entropy(
        self,
        guess_word: str,
        candidates: list[str],
        weights: dict[str, float],
    ) -> float:
        """
        H(guess) = -Σ_f  p(f|guess) · log₂(p(f|guess))

        p(f|guess) = Σ_{w : feedback(w,guess)=f}  weights[w]

        En uniform: cuenta palabras por grupo (entropía clásica).
        En frequency: las palabras frecuentes pesan más → el opener óptimo
                      prioriza separar bien las palabras comunes.

        Solo se usa en _compute_opener(). En guess() se usa _expected_score()
        que es más general (incorpora p_win y f̂).
        """
        partition: dict[tuple, float] = defaultdict(float)

        for w in candidates:
            pat = feedback(w, guess_word)
            partition[pat] += weights.get(w, 0.0)

        total = sum(partition.values())
        if total == 0:
            return 0.0

        h = 0.0
        for mass in partition.values():
            p = mass / total
            if p > 0.0:
                h -= p * math.log2(p)

        return h

    # ──────────────────────────────────────────────────────────────────────────
    # Generación de no-palabras
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_non_words(self) -> list[str]:
        """
        No-palabras informativas: combinaciones de las letras más frecuentes
        sin restricciones fonológicas del español.

        Ventaja vs palabras reales:
          Las palabras válidas tienen restricciones (ej: no puede haber ciertas
          combinaciones de consonantes). Una no-palabra puede poner las wl
          letras más frecuentes exactamente en sus posiciones óptimas.

        Costo: p_win = 0 (nunca ganan directamente).
          Válidas solo cuando p_win es pequeño de todas formas:
          opener y turno 2 con muchos candidatos.

        Algoritmo:
          1. Frecuencia ponderada de cada letra por posición sobre vocab completo.
          2. Top letras globales.
          3. Combinaciones C(top, wl) sin repetición.
          4. Para cada combo: asignación greedy (letra → posición de mayor freq).
          5. Filtrar las que ya son palabras del vocabulario.
        """
        wl      = self._wl
        vocab   = self._vocab
        weights = self._vocab_weights

        # Frecuencia ponderada letra×posición
        pos_freq: list[dict[str, float]] = [defaultdict(float) for _ in range(wl)]
        for w in vocab:
            w_weight = weights.get(w, 1.0 / max(len(vocab), 1))
            for i, ch in enumerate(w):
                pos_freq[i][ch] += w_weight

        # Frecuencia global
        overall: dict[str, float] = defaultdict(float)
        for i in range(wl):
            for ch, freq in pos_freq[i].items():
                overall[ch] += freq

        top_letters = [ch for ch, _ in sorted(overall.items(), key=lambda x: -x[1])]

        # Pool de letras: wl+4 para variedad sin explotar combinatoriamente
        # C(8,4)=70, C(9,5)=126, C(10,6)=210 — manejable
        pool_size    = min(wl + 4, len(top_letters))
        cand_letters = top_letters[:pool_size]

        non_words: list[str] = []
        seen: set[str]       = set()

        for combo in itertools.combinations(cand_letters, wl):
            remaining  = list(combo)
            assignment = [''] * wl

            for i in range(wl):
                best_ch = max(remaining, key=lambda ch: pos_freq[i].get(ch, 0.0))
                assignment[i] = best_ch
                remaining.remove(best_ch)

            nw = "".join(assignment)

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
        Opener óptimo = argmax_g H(g) evaluado contra el vocabulario COMPLETO.

        El vocabulario del lado "secreto" NUNCA se muestrea — hacerlo
        distorsionaría la entropía real. Solo el pool de openers se limita.

        Pool de evaluación:
          - Top MAX_OPENER_VOCAB_POOL palabras por número de letras únicas.
            Razón: más letras únicas = más hipótesis testeadas simultáneamente.
            Una palabra con 5 letras distintas descarta/confirma 5 posiciones;
            una con 3 únicas y 2 repetidas solo aporta información de 3.
          - Hasta MAX_NON_WORDS no-palabras generadas por _generate_non_words().

        Complejidad: O((MAX_OPENER_VOCAB_POOL + MAX_NON_WORDS) × |vocab|)
        """
        vocab   = self._vocab
        weights = self._vocab_weights

        vocab_pool = sorted(vocab, key=lambda w: len(set(w)), reverse=True)
        vocab_pool = vocab_pool[:MAX_OPENER_VOCAB_POOL]

        opener_pool  = vocab_pool + self._non_words

        best_guess   = vocab[0]
        best_entropy = -1.0

        for g in opener_pool:
            h = self._compute_entropy(g, vocab, weights)
            if h > best_entropy:
                best_entropy = h
                best_guess   = g

        return best_guess

    # ──────────────────────────────────────────────────────────────────────────
    # Expected score
    # ──────────────────────────────────────────────────────────────────────────

    def _expected_score(
        self,
        guess_word: str,
        candidates: list[str],
        weights: dict[str, float],
    ) -> float:
        """
        Costo esperado total (en guesses) de usar guess_word en el estado actual.

        Score(g) = Σ_f  p(f|g) × cost(f)

          cost(all-green) = 1                  ← ganamos en este turno
          cost(otro f)    = 1 + f̂(|C_f|)      ← 1 gastado + los que faltan

        p(f|g) = Σ_{w ∈ C_f} weights[w]

        Por qué es mejor que entropía pura:
          La entropía maximiza información pero ignora p_win.
          El expected score incorpora ambos: cuando p_win es alto
          (candidato muy probable), puede valer la pena adivinar directamente
          aunque des menos información en caso de fallo.

        Objetivo: MINIMIZAR este valor.
        """
        win_pat = tuple([2] * self._wl)

        partition: dict[tuple, list[str]] = defaultdict(list)
        for w in candidates:
            pat = feedback(w, guess_word)
            partition[pat].append(w)

        total_weight = sum(weights.get(w, 0.0) for w in candidates)
        if total_weight == 0.0:
            total_weight = 1.0

        score = 0.0
        for pat, group in partition.items():
            group_weight = sum(weights.get(w, 0.0) for w in group)
            p_f = group_weight / total_weight

            if pat == win_pat:
                score += p_f * 1.0
            else:
                score += p_f * (1.0 + _f_hat(len(group)))

        return score

    # ──────────────────────────────────────────────────────────────────────────
    # guess — lógica de decisión por turno
    # ──────────────────────────────────────────────────────────────────────────

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        """
        Selecciona el siguiente guess dado el historial de (guess, feedback).

        Turno 1 (history vacío):
            → Opener cacheado. Instantáneo (O(1)).

        Turno 2+:
            1. Filtrar candidatos consistentes con todo el feedback acumulado.
            2. Casos triviales:
               1 candidato → devolverlo.
               2 candidatos → devolver el más probable (óptimo siempre).
            3. Renormalizar pesos sobre los candidatos actuales.
            4. Construir pool de evaluación:
               - Base: todos los candidatos (tienen p_win > 0).
               - Turno 2, ≥30 candidatos: añadir no-palabras del caché.
               - Cap: MAX_GUESS_POOL (priorizar candidatos sobre no-palabras).
            5. Evaluar _expected_score para cada g en el pool.
            6. Devolver el g con menor score.
        """

        # ── Turno 1: opener cacheado ──────────────────────────────────────────
        if not history:
            return self._opener

        # ── Filtrar candidatos ─────────────────────────────────────────────────
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            return self._vocab[0]  # fallback de seguridad

        # ── Casos triviales ────────────────────────────────────────────────────
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) == 2:
            return max(candidates, key=lambda w: self._probs.get(w, 0.0))

        # ── Pesos renormalizados ───────────────────────────────────────────────
        weights = self._normalize_weights(candidates)

        # ── Pool de evaluación ────────────────────────────────────────────────
        turn = len(history) + 1

        guess_pool = list(candidates)

        if turn == 2 and len(candidates) >= NON_WORD_THRESHOLD_TURN2:
            non_words  = getattr(self, '_non_words', [])
            guess_pool = list(candidates) + non_words[:MAX_NON_WORDS_TURN2]

        if len(guess_pool) > MAX_GUESS_POOL:
            cands_set   = set(candidates)
            non_cands   = [g for g in guess_pool if g not in cands_set]
            extra_slots = max(0, MAX_GUESS_POOL - len(candidates))
            guess_pool  = candidates + non_cands[:extra_slots]

        # ── Evaluar y devolver el mínimo ──────────────────────────────────────
        best_guess = candidates[0]
        best_score = float('inf')

        for g in guess_pool:
            s = self._expected_score(g, candidates, weights)
            if s < best_score:
                best_score = s
                best_guess = g

        return best_guess