"""
ESTRATEGIA HÍBRIDA ÓPTIMA — Gabriel & Regina
══════════════════════════════════════════════════════════════════════

ARQUITECTURA POR TURNO
─────────────────────
T1: Opener precomputado (búsqueda exhaustiva 27^wl, no-palabra óptima)
T2: Tabla precomputada — lookup O(1) por pat_T1
    Selección exacta por entropía exhaustiva + tiebreaker jerarquizado
    + recomendación direct/probe por modo
T3: Tabla precomputada — lookup O(1) por (pat_T1|pat_T2)
    Clasificación: trivial/two/cluster1/cluster2/few/many
    E[direct] vs E[probe] exacto para ≤12 candidatos
T4: Runtime — objetivo cambia cualitativamente
    Prioridad 1: safe guess (garantiza P(fallo)=0)
    Prioridad 2: E[direct] vs E[probe] exacto
    Prioridad 3: minimax / entropía
T5: Runtime — penúltimo intento
    safe guess con max_group=1, o maximizar P(win en T5 o T6)
T6: Siempre el candidato más probable (regla matemáticamente rigurosa)

CARGA DE TABLAS
───────────────
Las tablas JSON se cargan a nivel clase (una sola vez por proceso).
begin_game() las accede sin I/O adicional en juegos subsecuentes.
Ruta: 3 niveles arriba del strategy.py hacia el root del repo.

FALLBACKS
──────────
Si una tabla no existe, cada turno tiene lógica runtime de fallback:
T2 → entropía exhaustiva sobre vocabulario
T3 → clasificación runtime + E exacto
T4-T6 → siempre tienen lógica runtime (no dependen de tablas)

RESTRICCIONES DEL TORNEO
─────────────────────────
5 segundos por juego (begin_game + todos los guess)
1 core de CPU
numpy + stdlib únicamente
allow_non_words=True (se pueden usar no-palabras)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import feedback as framework_feedback, filter_candidates

# ─── Constantes ──────────────────────────────────────────────────────────────

SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")
CHAR_TO_IDX     = {ch: i for i, ch in enumerate(SPANISH_LETTERS)}

OPENERS = {
    4: {"uniform": "aore", "frequency": "aore"},
    5: {"uniform": "careo", "frequency": "careo"},
    6: {"uniform": "ceriao", "frequency": "ceriao"},
}

# Umbral de candidatos para buscar safe guess en T4
T4_SAFE_GUESS_MAX_CANDS   = 10
# Umbral de probabilidad directa en T4 (frequency mode)
T4_DIRECT_PROB_THRESHOLD  = 0.50
# Umbral de probabilidad directa en T3 fallback (frequency mode)
T3_DIRECT_PROB_THRESHOLD  = 0.55

# Directorio raíz del repo (3 niveles arriba de este archivo)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Feedback vectorizado (para búsquedas exhaustivas en T2/T3 fallback) ─────

def _encode_words_numpy(words, wl):
    mat = np.zeros((len(words), wl), dtype=np.int8)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            mat[i, j] = CHAR_TO_IDX.get(ch, 0)
    return mat


def _feedbacks_numpy(guess_enc, secrets_enc, wl):
    """Feedback batch: guess_enc (wl,) contra secrets_enc (N, wl) → (N,) int."""
    N       = secrets_enc.shape[0]
    greens  = (secrets_enc == guess_enc[np.newaxis, :])
    yellows = np.zeros((N, wl), dtype=bool)
    for i in range(wl):
        if greens[:, i].all():
            continue
        guess_ch    = guess_enc[i]
        not_green_i = ~greens[:, i]
        available   = (secrets_enc == guess_ch) & ~greens
        consumed    = np.zeros(N, dtype=np.int32)
        for j in range(i):
            if guess_enc[j] == guess_ch:
                consumed += (greens[:, j] | yellows[:, j]).astype(np.int32)
        yellows[:, i] = not_green_i & (available.sum(axis=1) > consumed)
    pat_mat = np.zeros((N, wl), dtype=np.int32)
    pat_mat[greens]  = 2
    pat_mat[yellows] = 1
    powers = np.array([3 ** j for j in range(wl - 1, -1, -1)], dtype=np.int32)
    return (pat_mat * powers[np.newaxis, :]).sum(axis=1)


def _entropy(feedbacks, weights_arr, n_patterns):
    pat_w = np.bincount(feedbacks, weights=weights_arr, minlength=n_patterns)
    mask  = pat_w > 0
    p     = pat_w[mask]
    return float(-np.sum(p * np.log2(p)))


# ─── Utilidades de candidatos ─────────────────────────────────────────────────

def _normalize_weights(candidates, probs_dict):
    raw   = [probs_dict.get(w, 1e-10) for w in candidates]
    total = sum(raw)
    return [v / total for v in raw]


def _most_probable(candidates, probs_dict):
    return max(candidates, key=lambda w: probs_dict.get(w, 0.0))


def _best_prob(candidates, probs_dict):
    return probs_dict.get(_most_probable(candidates, probs_dict), 0.0)


# ─── Safe guess finder ───────────────────────────────────────────────────────

def _is_safe_guess(guess, candidates, wl, max_group_size=2):
    """
    Verifica si guess garantiza que todos los grupos resultantes
    tienen ≤ max_group_size candidatos.
    Usa framework_feedback para exactitud absoluta.
    """
    groups = defaultdict(int)
    for c in candidates:
        pat = framework_feedback(c, guess)
        groups[pat] += 1
        if groups[pat] > max_group_size:
            return False
    return True


def _find_safe_guess(candidates, vocab, wl, max_group_size=2, probs_dict=None):
    """
    Busca un safe guess que garantice todos los grupos ≤ max_group_size.

    Orden de búsqueda:
    1. Candidatos activos primero (si son safe, ganamos directamente)
    2. Vocabulario completo
    3. No-palabras candidatas (strings de letras nuevas)

    Retorna: (guess, True) si encontró safe, (None, False) si no.
    """
    # Paso 1: candidatos activos primero
    # Un candidato que es safe es mejor — puede ganar este turno
    cands_set = set(candidates)

    # Ordenar candidatos por probabilidad descendente para preferir
    # el más probable en caso de múltiples safe guesses
    ordered_cands = sorted(candidates,
                            key=lambda w: -(probs_dict or {}).get(w, 0.0))
    for g in ordered_cands:
        if _is_safe_guess(g, candidates, wl, max_group_size):
            return g, True

    # Paso 2: vocabulario completo
    for g in vocab:
        if g in cands_set:
            continue  # ya probado
        if _is_safe_guess(g, candidates, wl, max_group_size):
            return g, True

    return None, False


# ─── Cálculo exacto de E[direct] ─────────────────────────────────────────────

def _expected_cost_direct(candidates, probs_dict, turns_left):
    """
    E[adivinar el más probable en este turno].

    E = p_best × 1 + (1-p_best) × (1 + E*[n-1, turns_left-1])

    Con turns_left restantes, si no adivinamos este turno, quedan
    turns_left-1 para n-1 candidatos. La cota de E*:
    - 1 candidato: E*=1
    - 2 candidatos: E*=1.5
    - ≥3 y turns≥2: E*=log₂(n-1)+1 (cota optimista)
    - ≥3 y turns=1: E*=inf (fallo garantizado)
    """
    n = len(candidates)
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0

    raw   = [probs_dict.get(w, 1e-10) for w in candidates]
    total = sum(raw)
    p_best = max(raw) / total

    n_remaining = n - 1
    if n_remaining == 0:
        return 1.0
    if n_remaining == 1:
        e_remaining = 1.0
    elif turns_left <= 1:
        e_remaining = float('inf')
    elif n_remaining == 2:
        e_remaining = 1.5
    else:
        e_remaining = math.log2(n_remaining) + 1.0

    return p_best * 1.0 + (1.0 - p_best) * (1.0 + e_remaining)


# ─── Estrategia T4 ───────────────────────────────────────────────────────────

def _choose_t4(candidates, vocab, wl, mode, probs_dict):
    """
    T4 — Pivote crítico. Con 3 intentos restantes (T4, T5, T6).

    Lógica jerarquizada:
    1. ≤2 candidatos: adivinar el más probable directamente
    2. 3-10 candidatos: buscar safe guess (garantiza P(fallo)=0)
       - Si existe: usarlo (aunque no sea el de mayor entropía)
       - En frequency: comparar E[safe] vs E[direct], elegir menor
    3. >10 candidatos sin safe guess: entropía o direct según modo
    """
    n = len(candidates)

    # Caso 1: trivial
    if n <= 2:
        return _most_probable(candidates, probs_dict)

    # Caso 2: buscar safe guess
    if n <= T4_SAFE_GUESS_MAX_CANDS:
        safe_g, found = _find_safe_guess(candidates, vocab, wl,
                                          max_group_size=2,
                                          probs_dict=probs_dict)
        if found:
            # En frequency mode: verificar que safe_g es mejor que adivinar directo
            if mode == "frequency":
                e_direct = _expected_cost_direct(candidates, probs_dict,
                                                  turns_left=3)
                # E[safe_guess] ≈ 2 (siempre resuelve en ≤2 intentos más)
                # E[direct] puede ser menor si p_best es muy alto
                p_best = _best_prob(candidates, probs_dict)
                e_safe = (1.0 - p_best) * 2.0 + p_best * 1.0  # si safe_g es candidato
                # Si safe_g no es candidato, E[safe] ≈ 2.0 siempre
                if safe_g not in set(candidates):
                    e_safe = 2.0
                if e_direct < e_safe - 1e-9:
                    return _most_probable(candidates, probs_dict)
            return safe_g

    # Caso 3: muchos candidatos o no hay safe guess
    if mode == "frequency":
        p_best = _best_prob(candidates, probs_dict)
        if p_best > T4_DIRECT_PROB_THRESHOLD:
            return _most_probable(candidates, probs_dict)

    # Fallback: mejor por entropía sobre el vocabulario (capped a 300 palabras)
    return _best_entropy_guess_vocab(candidates, vocab, wl, probs_dict,
                                      max_pool=300)


# ─── Estrategia T5 ───────────────────────────────────────────────────────────

def _choose_t5(candidates, vocab, wl, mode, probs_dict):
    """
    T5 — Penúltimo intento. Con 2 intentos restantes (T5, T6).

    Lógica:
    1. 1-2 candidatos: adivinar el más probable
    2. 3 candidatos: buscar safe guess con max_group=1
       - Si existe: P(fallo)=0
       - Si no: adivinar el más probable (P(fallo) = 1-p_best)
    3. ≥4 candidatos: maximizar P(ganar en T5 o T6)
       argmax_g Σ_f p(f|g,S) × 1[|grupo_f| ≤ 1]
    """
    n = len(candidates)

    # Caso 1: trivial
    if n <= 2:
        return _most_probable(candidates, probs_dict)

    # Caso 2: 3 candidatos — buscar safe guess con max_group=1
    if n == 3:
        safe_g, found = _find_safe_guess(candidates, vocab, wl,
                                          max_group_size=1,
                                          probs_dict=probs_dict)
        if found:
            return safe_g
        return _most_probable(candidates, probs_dict)

    # Caso 3: ≥4 candidatos — maximizar P(win en T5 o T6)
    # P(win) = Σ_f p(f|g,S) × 1[|grupo_f| ≤ 1]
    # Esto incluye: grupo_f vacío (guess acertó = green total) o tamaño 1
    return _best_win_probability_guess(candidates, vocab, wl, probs_dict,
                                        max_pool=300)


def _best_win_probability_guess(candidates, vocab, wl, probs_dict, max_pool=300):
    """
    Encuentra el guess que maximiza P(ganar en T5 o T6):
      P(win) = Σ_f p(f|g,S) × 1[|grupo_f| ≤ 1]

    Grupos de tamaño 0: el guess acertó (grupo victoria)
    Grupos de tamaño 1: T6 garantiza victoria
    Grupos de tamaño ≥2: P(fallo) > 0

    Candidatos activos se buscan primero (pueden ganar en T5).
    """
    raw      = [probs_dict.get(c, 1e-10) for c in candidates]
    total_w  = sum(raw)
    probs_n  = [v / total_w for v in raw]

    # Pool: candidatos primero, luego vocab limitado
    pool = list(candidates)
    remaining_budget = max_pool - len(pool)
    cands_set = set(candidates)
    for w in vocab:
        if w not in cands_set and remaining_budget > 0:
            pool.append(w)
            remaining_budget -= 1

    best_guess  = _most_probable(candidates, probs_dict)
    best_p_win  = 0.0

    for g in pool:
        groups = defaultdict(float)
        win_pat = tuple([2] * wl)
        for c, p in zip(candidates, probs_n):
            pat = framework_feedback(c, g)
            groups[pat] += p

        p_win = 0.0
        for pat, p_f in groups.items():
            # Reconstruir tamaño del grupo
            group_size = sum(1 for c in candidates
                             if framework_feedback(c, g) == pat)
            if group_size <= 1:
                p_win += p_f

        if p_win > best_p_win + 1e-9:
            best_p_win  = p_win
            best_guess  = g
        elif abs(p_win - best_p_win) <= 1e-9:
            # Tiebreaker: preferir candidato activo
            if g in cands_set and best_guess not in cands_set:
                best_guess = g

    return best_guess


# ─── Búsqueda de entropía sobre vocabulario ──────────────────────────────────

def _best_entropy_guess_vocab(candidates, vocab, wl, probs_dict, max_pool=500):
    """
    Encuentra el guess de máxima entropía sobre un pool de vocabulario.
    Usa NumPy para evaluación en batch.
    Tiebreaker: preferir candidato activo.
    """
    raw      = [probs_dict.get(c, 1e-10) for c in candidates]
    total    = sum(raw)
    w_arr    = np.array([v / total for v in raw], dtype=np.float64)
    cands_enc = _encode_words_numpy(candidates, wl)
    n_patterns = 3 ** wl

    # Pool de guesses
    cands_set = set(candidates)
    pool      = list(candidates)
    remaining = max_pool - len(pool)
    for w in vocab:
        if w not in cands_set and remaining > 0:
            pool.append(w)
            remaining -= 1

    best_H    = -1.0
    best_g    = candidates[0]
    best_cand = candidates[0] in cands_set

    pool_enc = _encode_words_numpy(pool, wl)
    for i, g in enumerate(pool):
        fbs = _feedbacks_numpy(pool_enc[i], cands_enc, wl)
        H   = _entropy(fbs, w_arr, n_patterns)
        g_is_cand = g in cands_set
        if H > best_H + 1e-9:
            best_H    = H
            best_g    = g
            best_cand = g_is_cand
        elif abs(H - best_H) <= 1e-9 and not best_cand and g_is_cand:
            best_g    = g
            best_cand = True

    return best_g


# ─── Fallback T3 runtime ─────────────────────────────────────────────────────

def _choose_t3_runtime(candidates, vocab, wl, mode, probs_dict):
    """
    Fallback para T3 cuando el estado no está en la tabla precomputada.
    Replica la lógica del precompute_t3 pero en runtime.

    Clasifica el estado y elige la acción óptima.
    """
    n = len(candidates)

    if n <= 1:
        return candidates[0] if candidates else vocab[0]

    if n == 2:
        return _most_probable(candidates, probs_dict)

    # Frequency mode: considerar adivinar directamente si p_best alto
    if mode == "frequency":
        p_best = _best_prob(candidates, probs_dict)
        if p_best > T3_DIRECT_PROB_THRESHOLD:
            return _most_probable(candidates, probs_dict)

    # Buscar entropía exhaustiva sobre vocabulario (pool capped)
    return _best_entropy_guess_vocab(candidates, vocab, wl, probs_dict,
                                      max_pool=400)


# ─── Clase principal ─────────────────────────────────────────────────────────

class GabrielReginaStrategy(Strategy):
    """
    Estrategia híbrida óptima para el torneo de Wordle.

    Combina tablas precomputadas (T1-T3) con lógica runtime adaptativa
    (T4-T6) para maximizar la tasa de éxito y minimizar el promedio de
    intentos en los 6 escenarios del torneo.
    """

    # ─── Caché de clase: tablas cargadas una vez por proceso ─────────────────
    _t2_tables: dict[str, dict] = {}
    _t3_tables: dict[str, dict] = {}
    _tables_loaded: set[str]    = set()

    @property
    def name(self) -> str:
        return "HybridOptimal_gabriel_regina"

    # ─── Carga de tablas ─────────────────────────────────────────────────────

    @classmethod
    def _load_tables(cls, wl: int, mode: str) -> None:
        """Carga las tablas T2 y T3 para (wl, mode) si no están en caché."""
        key = f"{wl}_{mode}"
        if key in cls._tables_loaded:
            return

        # Buscar en el root del repo (3 niveles arriba de este archivo)
        repo_root = Path(__file__).resolve().parent.parent.parent

        # T2
        t2_path = repo_root / f"t2_table_{wl}_{mode}.json"
        if t2_path.exists():
            try:
                with open(t2_path, encoding='utf-8') as f:
                    cls._t2_tables[key] = json.load(f)
            except Exception:
                cls._t2_tables[key] = {}
        else:
            cls._t2_tables[key] = {}

        # T3
        t3_path = repo_root / f"t3_table_{wl}_{mode}.json"
        if t3_path.exists():
            try:
                with open(t3_path, encoding='utf-8') as f:
                    cls._t3_tables[key] = json.load(f)
            except Exception:
                cls._t3_tables[key] = {}
        else:
            cls._t3_tables[key] = {}

        cls._tables_loaded.add(key)

    # ─── begin_game ──────────────────────────────────────────────────────────

    def begin_game(self, config: GameConfig) -> None:
        self._wl      = config.word_length
        self._mode    = config.mode
        self._vocab   = list(config.vocabulary)
        self._probs   = dict(config.probabilities)
        self._opener  = OPENERS[self._wl][self._mode]
        self._max_g   = config.max_guesses  # típicamente 6

        # Cargar tablas (no-op si ya están en caché)
        GabrielReginaStrategy._load_tables(self._wl, self._mode)

        self._t2 = GabrielReginaStrategy._t2_tables.get(
            f"{self._wl}_{self._mode}", {})
        self._t3 = GabrielReginaStrategy._t3_tables.get(
            f"{self._wl}_{self._mode}", {})

    # ─── guess ───────────────────────────────────────────────────────────────

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        turn = len(history) + 1  # turno actual (1-indexed)

        # Filtrar candidatos con el feedback acumulado
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            return self._vocab[0]

        if len(candidates) == 1:
            return candidates[0]

        wl    = self._wl
        mode  = self._mode
        vocab = self._vocab
        probs = self._probs

        # ─── T1: Opener ───────────────────────────────────────────────────────
        if turn == 1:
            return self._opener

        # ─── T2: Lookup tabla ────────────────────────────────────────────────
        if turn == 2:
            pat_t1 = ''.join(str(x) for x in history[0][1])
            guess  = self._t2.get(pat_t1)
            if guess:
                return guess
            # Fallback: entropía sobre vocabulario
            return _best_entropy_guess_vocab(candidates, vocab, wl, probs,
                                              max_pool=500)

        # ─── T3: Lookup tabla ────────────────────────────────────────────────
        if turn == 3:
            pat_t1  = ''.join(str(x) for x in history[0][1])
            pat_t2  = ''.join(str(x) for x in history[1][1])
            state   = f"{pat_t1}|{pat_t2}"
            entry   = self._t3.get(state)
            if entry:
                g = entry.get('guess')
                if g:
                    return g
            # Fallback: lógica runtime
            return _choose_t3_runtime(candidates, vocab, wl, mode, probs)

        # ─── T4: Safe guess o estrategia adaptativa ──────────────────────────
        if turn == 4:
            return _choose_t4(candidates, vocab, wl, mode, probs)

        # ─── T5: Penúltimo ───────────────────────────────────────────────────
        if turn == 5:
            return _choose_t5(candidates, vocab, wl, mode, probs)

        # ─── T6+: Siempre el más probable (regla matemáticamente rigurosa) ───
        return _most_probable(candidates, probs)