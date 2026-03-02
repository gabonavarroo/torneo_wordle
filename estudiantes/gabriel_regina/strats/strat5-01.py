"""
ESTRATEGIA HÍBRIDA ÓPTIMA — Gabriel & Regina  (v2 — fixes aplicados)
══════════════════════════════════════════════════════════════════════

ARQUITECTURA POR TURNO
─────────────────────
T1: Opener precomputado — alineado exactamente con las tablas T2/T3
T2: Tabla precomputada — lookup O(1) por pat_T1
T3: Tabla precomputada — lookup O(1) por (pat_T1|pat_T2)
    Fallback mejorado con pool adaptativo según vocab size
T4: Runtime — safe guess (P(fallo)=0) o E exacto o entropía
    Pool adaptativo: 500/4L, 800/5L, 1000/6L
T5: Runtime — maximizar P(win en T5|T6) con pool adaptativo
T6: Siempre el candidato más probable

CORRECCIONES RESPECTO A v1
───────────────────────────
BUG 1 FIJO: Openers sincronizados con tablas precomputadas
  5L uniform/frequency: 'careo'   (era 'sareo' — MISMATCH critico)
  6L uniform:           'carieo'  (era 'ceriao' — MISMATCH critico)
  4L y 6L frequency: sin cambio ('aore', 'carieo')

BUG 2 FIJO: Pool T3 fallback escalado por vocabulario
  4L: 1000  (era 400)
  5L: 2000  (era 400 — solo cubria el 9% del vocab!)
  6L: 2500  (era 400 — solo cubria el 7% del vocab!)

BUG 3 FIJO: Pool T4 entropía escalado por vocabulario
  4L: 500, 5L: 800, 6L: 1000  (era 300 para todos)

BUG 4 FIJO: _best_win_probability_guess O(n^2) → O(n)
  El calculo de group_size relanzaba framework_feedback innecesariamente.
  Ahora usa una sola pasada por candidatos y acumula (prob, count).

MEJORA 5: Pool T5 win-probability escalado
  4L: 400, 5L: 600, 6L: 800  (era 300 para todos)

MEJORA 6: T4 safe guess umbral subido a 12 candidatos (era 10)
  Con n<=12 la busqueda cuesta O(12 x |vocab|) aprox 0.03s — factible.

NOTA FUTURA: Cuando termine el nuevo run de 6L frequency con opener
'ceriao', actualizar OPENERS[6]['frequency'] = 'ceriao'.

RESTRICCIONES DEL TORNEO
─────────────────────────
5 segundos por juego | 1 core de CPU | numpy + stdlib
allow_non_words=True
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

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  OPENERS — DEBEN COINCIDIR con el opener usado en precompute_t2/t3.    │
# │  4L: 'aore'   → tablas 4L uniform + frequency                          │
# │  5L: 'sareo'  → tablas 5L uniform + frequency (recomputadas)           │
# │  6L: 'ceriao' → tablas 6L uniform + frequency (recomputadas)           │
# └─────────────────────────────────────────────────────────────────────────┘
OPENERS = {
    4: {"uniform": "aore",   "frequency": "aore"},
    5: {"uniform": "sareo",  "frequency": "sareo"},
    6: {"uniform": "ceriao", "frequency": "carieo"},
}

# Pool sizes adaptativos por wl
_T3_POOL = {4: 1000, 5: 2000, 6: 2500}
_T4_POOL = {4:  500, 5:  800, 6: 1000}
_T5_POOL = {4:  400, 5:  600, 6:  800}

# Umbral de candidatos para safe guess en T4
T4_SAFE_GUESS_MAX_CANDS  = 12
# Umbral de probabilidad directa en T4 (frequency mode)
T4_DIRECT_PROB_THRESHOLD = 0.50
# Umbral de probabilidad directa en T3 fallback (frequency mode)
T3_DIRECT_PROB_THRESHOLD = 0.55


# ─── Feedback vectorizado ────────────────────────────────────────────────────

def _encode_words_numpy(words, wl):
    mat = np.zeros((len(words), wl), dtype=np.int8)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            mat[i, j] = CHAR_TO_IDX.get(ch, 0)
    return mat


def _feedbacks_numpy(guess_enc, secrets_enc, wl):
    """Feedback batch: guess_enc (wl,) contra secrets_enc (N, wl) → (N,) int."""
    N      = secrets_enc.shape[0]
    greens = (secrets_enc == guess_enc[np.newaxis, :])
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

def _most_probable(candidates, probs_dict):
    return max(candidates, key=lambda w: probs_dict.get(w, 0.0))


def _best_prob(candidates, probs_dict):
    return probs_dict.get(_most_probable(candidates, probs_dict), 0.0)


# ─── Safe guess finder ───────────────────────────────────────────────────────

def _is_safe_guess(guess, candidates, wl, max_group_size=2):
    """True si todos los grupos de (candidates, guess) tienen <= max_group_size."""
    groups = defaultdict(int)
    for c in candidates:
        pat = framework_feedback(c, guess)
        groups[pat] += 1
        if groups[pat] > max_group_size:
            return False
    return True


def _find_safe_guess(candidates, vocab, wl, max_group_size=2, probs_dict=None):
    """
    Busca un safe guess garantizando todos los grupos <= max_group_size.
    Preferencia: candidatos activos (por prob desc) → vocab completo.
    """
    cands_set = set(candidates)
    ordered_cands = sorted(candidates,
                           key=lambda w: -(probs_dict or {}).get(w, 0.0))
    for g in ordered_cands:
        if _is_safe_guess(g, candidates, wl, max_group_size):
            return g, True
    for g in vocab:
        if g in cands_set:
            continue
        if _is_safe_guess(g, candidates, wl, max_group_size):
            return g, True
    return None, False


# ─── Cálculo exacto de E[direct] ─────────────────────────────────────────────

def _expected_cost_direct(candidates, probs_dict, turns_left):
    """
    E[adivinar el mas probable ahora].
    p_best x 1 + (1-p_best) x (1 + E*[n-1, turns_left-1])
    """
    n = len(candidates)
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0
    raw    = [probs_dict.get(w, 1e-10) for w in candidates]
    total  = sum(raw)
    p_best = max(raw) / total
    n_rem  = n - 1
    if n_rem == 0:
        return 1.0
    if n_rem == 1:
        e_rem = 1.0
    elif turns_left <= 1:
        e_rem = float('inf')
    elif n_rem == 2:
        e_rem = 1.5
    else:
        e_rem = math.log2(n_rem) + 1.0
    return p_best + (1.0 - p_best) * (1.0 + e_rem)


# ─── Estrategia T4 ───────────────────────────────────────────────────────────

def _choose_t4(candidates, vocab, wl, mode, probs_dict):
    """
    T4 — 3 intentos restantes (T4, T5, T6).
    1. n<=2: mas probable directo
    2. 3<=n<=12: safe guess (P(fallo)=0); en frequency comparar E[direct] vs E[safe]
    3. n>12: entropia escalada o direct si p_best alto (frequency)
    """
    n = len(candidates)

    if n <= 2:
        return _most_probable(candidates, probs_dict)

    if n <= T4_SAFE_GUESS_MAX_CANDS:
        safe_g, found = _find_safe_guess(candidates, vocab, wl,
                                          max_group_size=2,
                                          probs_dict=probs_dict)
        if found:
            if mode == "frequency":
                e_direct = _expected_cost_direct(candidates, probs_dict, turns_left=3)
                p_best   = _best_prob(candidates, probs_dict)
                e_safe   = (1.0 - p_best) * 2.0 + p_best * 1.0
                if safe_g not in set(candidates):
                    e_safe = 2.0
                if e_direct < e_safe - 1e-9:
                    return _most_probable(candidates, probs_dict)
            return safe_g

    if mode == "frequency":
        if _best_prob(candidates, probs_dict) > T4_DIRECT_PROB_THRESHOLD:
            return _most_probable(candidates, probs_dict)

    max_pool = _T4_POOL.get(wl, 500)
    return _best_entropy_guess_vocab(candidates, vocab, wl, probs_dict,
                                      max_pool=max_pool)


# ─── Estrategia T5 ───────────────────────────────────────────────────────────

def _choose_t5(candidates, vocab, wl, mode, probs_dict):
    """
    T5 — 2 intentos restantes (T5, T6).
    1. n<=2: mas probable
    2. n==3: safe guess max_group=1 si existe, sino mas probable
    3. n>=4: maximizar P(win en T5 o T6)
    """
    n = len(candidates)

    if n <= 2:
        return _most_probable(candidates, probs_dict)

    if n == 3:
        safe_g, found = _find_safe_guess(candidates, vocab, wl,
                                          max_group_size=1,
                                          probs_dict=probs_dict)
        if found:
            return safe_g
        return _most_probable(candidates, probs_dict)

    max_pool = _T5_POOL.get(wl, 400)
    return _best_win_probability_guess(candidates, vocab, wl, probs_dict,
                                        max_pool=max_pool)


def _best_win_probability_guess(candidates, vocab, wl, probs_dict, max_pool=400):
    """
    Maximiza P(win en este turno o el siguiente):
      P(win) = sum_f p(f|g,S) * 1[|grupo_f| <= 1]

    Complejidad: O(pool x n) — una sola pasada por candidatos por guess.
    """
    raw     = [probs_dict.get(c, 1e-10) for c in candidates]
    total_w = sum(raw)
    probs_n = [v / total_w for v in raw]

    pool = list(candidates)
    budget = max_pool - len(pool)
    cands_set = set(candidates)
    for w in vocab:
        if w not in cands_set and budget > 0:
            pool.append(w)
            budget -= 1

    best_guess = _most_probable(candidates, probs_dict)
    best_p_win = 0.0

    for g in pool:
        # Acumular (prob_sum, count) por patron — una sola pasada
        groups: dict = {}
        for c, p in zip(candidates, probs_n):
            pat = framework_feedback(c, g)
            if pat not in groups:
                groups[pat] = [0.0, 0]
            groups[pat][0] += p
            groups[pat][1] += 1

        p_win = sum(v[0] for v in groups.values() if v[1] <= 1)

        if p_win > best_p_win + 1e-9:
            best_p_win = p_win
            best_guess = g
        elif abs(p_win - best_p_win) <= 1e-9:
            if g in cands_set and best_guess not in cands_set:
                best_guess = g

    return best_guess


# ─── Búsqueda de entropía sobre vocabulario ──────────────────────────────────

def _best_entropy_guess_vocab(candidates, vocab, wl, probs_dict, max_pool=500):
    """
    Guess de maxima entropia sobre pool. NumPy batch. Tiebreaker: candidato activo.
    """
    raw       = [probs_dict.get(c, 1e-10) for c in candidates]
    total     = sum(raw)
    w_arr     = np.array([v / total for v in raw], dtype=np.float64)
    cands_enc = _encode_words_numpy(candidates, wl)
    n_patterns = 3 ** wl

    cands_set = set(candidates)
    pool      = list(candidates)
    remaining = max_pool - len(pool)
    for w in vocab:
        if w not in cands_set and remaining > 0:
            pool.append(w)
            remaining -= 1

    best_H    = -1.0
    best_g    = candidates[0]
    best_cand = True

    pool_enc = _encode_words_numpy(pool, wl)
    for i, g in enumerate(pool):
        fbs       = _feedbacks_numpy(pool_enc[i], cands_enc, wl)
        H         = _entropy(fbs, w_arr, n_patterns)
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
    Fallback T3 cuando el estado no esta en tabla.
    Pool escalado por wl para cubrir bien el espacio de guesses.
    """
    n = len(candidates)

    if n <= 1:
        return candidates[0] if candidates else vocab[0]
    if n == 2:
        return _most_probable(candidates, probs_dict)

    if mode == "frequency":
        p_best = _best_prob(candidates, probs_dict)
        if p_best > T3_DIRECT_PROB_THRESHOLD:
            return _most_probable(candidates, probs_dict)

    max_pool = _T3_POOL.get(wl, 1000)
    return _best_entropy_guess_vocab(candidates, vocab, wl, probs_dict,
                                      max_pool=max_pool)


# ─── Clase principal ─────────────────────────────────────────────────────────

class GabrielReginaStrategy(Strategy):
    """
    Estrategia hibrida optima v2 para el torneo de Wordle en espanol.

    T1-T3: lookup precomputado O(1) con tablas t2/t3.
    T4-T6: runtime adaptativo con pools escalados por vocabulario.
    Openers completamente alineados con las tablas precomputadas.
    """

    _t2_tables:     dict[str, dict] = {}
    _t3_tables:     dict[str, dict] = {}
    _tables_loaded: set[str]        = set()

    @property
    def name(self) -> str:
        return "HybridOptimal_gabriel_regina"

    @classmethod
    def _load_tables(cls, wl: int, mode: str) -> None:
        key = f"{wl}_{mode}"
        if key in cls._tables_loaded:
            return
        repo_root = Path(__file__).resolve().parent.parent.parent
        for prefix, store in [("t2", cls._t2_tables), ("t3", cls._t3_tables)]:
            path = repo_root / f"{prefix}_table_{wl}_{mode}.json"
            if path.exists():
                try:
                    with open(path, encoding='utf-8') as f:
                        store[key] = json.load(f)
                except Exception:
                    store[key] = {}
            else:
                store[key] = {}
        cls._tables_loaded.add(key)

    def begin_game(self, config: GameConfig) -> None:
        self._wl     = config.word_length
        self._mode   = config.mode
        self._vocab  = list(config.vocabulary)
        self._probs  = dict(config.probabilities)
        self._opener = OPENERS[self._wl][self._mode]
        self._max_g  = config.max_guesses
        GabrielReginaStrategy._load_tables(self._wl, self._mode)
        key      = f"{self._wl}_{self._mode}"
        self._t2 = GabrielReginaStrategy._t2_tables.get(key, {})
        self._t3 = GabrielReginaStrategy._t3_tables.get(key, {})

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        turn = len(history) + 1

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

        # T1 ─────────────────────────────────────────────────────────────────
        if turn == 1:
            return self._opener

        # T2 ─────────────────────────────────────────────────────────────────
        if turn == 2:
            pat_t1 = ''.join(str(x) for x in history[0][1])
            g = self._t2.get(pat_t1)
            if g:
                return g
            # Fallback con pool generoso
            return _best_entropy_guess_vocab(candidates, vocab, wl, probs,
                                              max_pool=_T3_POOL.get(wl, 1000))

        # T3 ─────────────────────────────────────────────────────────────────
        if turn == 3:
            pat_t1 = ''.join(str(x) for x in history[0][1])
            pat_t2 = ''.join(str(x) for x in history[1][1])
            state  = f"{pat_t1}|{pat_t2}"
            entry  = self._t3.get(state)
            if entry:
                g = entry.get('guess')
                if g:
                    return g
            return _choose_t3_runtime(candidates, vocab, wl, mode, probs)

        # T4 ─────────────────────────────────────────────────────────────────
        if turn == 4:
            return _choose_t4(candidates, vocab, wl, mode, probs)

        # T5 ─────────────────────────────────────────────────────────────────
        if turn == 5:
            return _choose_t5(candidates, vocab, wl, mode, probs)

        # T6+ ────────────────────────────────────────────────────────────────
        return _most_probable(candidates, probs)