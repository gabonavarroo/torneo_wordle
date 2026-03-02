"""
Híbrido RG5 — RG4 + cluster-buster liviano en fallback T3.

Cambios vs RG4:
- En T3 fallback, si n ≤ 8 y hay exactamente 1 o 2 posiciones variables
  (cluster pequeño), construye un buster de no-palabra y compara su
  expected_score contra el fallback normal; usa el mejor.
- Uniform mantiene entropía como fallback; frequency mantiene probes
  dinámicos + no-palabras y p_best>0.60 para direct.
- Safe-guess T4/T5 y tablas T2/T3 se mantienen igual.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import feedback as framework_feedback, filter_candidates


# ─── Constantes ──────────────────────────────────────────────────────────────

SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")
CHAR_TO_IDX = {ch: i for i, ch in enumerate(SPANISH_LETTERS)}

OPENERS = {
    4: {"uniform": "aore", "frequency": "aore"},
    5: {"uniform": "sareo", "frequency": "sareo"},
    6: {"uniform": "ceriao", "frequency": "carieo"},
}

T4_SAFE_GUESS_MAX_CANDS = 10
T4_DIRECT_PROB_THRESHOLD_FREQ = 0.60
T4_DIRECT_PROB_THRESHOLD_UNI = 0.50

T3_DIRECT_PROB_THRESHOLD_FREQ = 0.60
T3_DIRECT_PROB_THRESHOLD_UNI = 0.55

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Feedback vectorizado ────────────────────────────────────────────────────

def _encode_words_numpy(words: Sequence[str], wl: int) -> np.ndarray:
    mat = np.zeros((len(words), wl), dtype=np.int8)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            mat[i, j] = CHAR_TO_IDX.get(ch, 0)
    return mat


def _feedbacks_numpy(guess_enc: np.ndarray, secrets_enc: np.ndarray, wl: int) -> np.ndarray:
    greens = secrets_enc == guess_enc[np.newaxis, :]
    yellows = np.zeros_like(greens)
    for i in range(wl):
        if greens[:, i].all():
            continue
        guess_ch = guess_enc[i]
        not_green = ~greens[:, i]
        available = (secrets_enc == guess_ch) & ~greens
        consumed = np.zeros(secrets_enc.shape[0], dtype=np.int32)
        for j in range(i):
            if guess_enc[j] == guess_ch:
                consumed += (greens[:, j] | yellows[:, j]).astype(np.int32)
        yellows[:, i] = not_green & (available.sum(axis=1) > consumed)
    pat = np.zeros((secrets_enc.shape[0], wl), dtype=np.int32)
    pat[greens] = 2
    pat[yellows] = 1
    powers = np.array([3 ** j for j in range(wl - 1, -1, -1)], dtype=np.int32)
    return (pat * powers[np.newaxis, :]).sum(axis=1)


def _entropy(feedbacks: np.ndarray, weights_arr: np.ndarray, n_patterns: int) -> float:
    pat_w = np.bincount(feedbacks, weights=weights_arr, minlength=n_patterns)
    mask = pat_w > 0
    p = pat_w[mask]
    return float(-np.sum(p * np.log2(p)))


# ─── Utilidades de candidatos ───────────────────────────────────────────────

def _normalize_weights(candidates: Sequence[str], probs_dict: dict[str, float]) -> dict[str, float]:
    raw = {w: probs_dict.get(w, 1e-10) for w in candidates}
    tot = sum(raw.values())
    if tot <= 0:
        u = 1.0 / len(candidates)
        return {w: u for w in candidates}
    return {w: v / tot for w, v in raw.items()}


def _most_probable(candidates: Sequence[str], probs_dict: dict[str, float]) -> str:
    return max(candidates, key=lambda w: probs_dict.get(w, 0.0))


def _best_prob(candidates: Sequence[str], probs_dict: dict[str, float]) -> float:
    return probs_dict.get(_most_probable(candidates, probs_dict), 0.0)


# ─── Safe guess finder ──────────────────────────────────────────────────────

def _is_safe_guess(guess: str, candidates: Sequence[str], max_group_size: int) -> bool:
    groups = defaultdict(int)
    for c in candidates:
        pat = framework_feedback(c, guess)
        groups[pat] += 1
        if groups[pat] > max_group_size:
            return False
    return True


def _find_safe_guess(candidates: Sequence[str], vocab: Sequence[str], max_group_size: int,
                     probs_dict: dict[str, float]) -> tuple[str | None, bool]:
    cands_set = set(candidates)
    ordered = sorted(candidates, key=lambda w: -probs_dict.get(w, 0.0))
    for g in ordered:
        if _is_safe_guess(g, candidates, max_group_size):
            return g, True
    for g in vocab:
        if g in cands_set:
            continue
        if _is_safe_guess(g, candidates, max_group_size):
            return g, True
    return None, False


# ─── Entropía sobre vocab ───────────────────────────────────────────────────

def _best_entropy_guess_vocab(candidates: Sequence[str], vocab: Sequence[str], wl: int,
                              probs_dict: dict[str, float], max_pool: int = 500) -> str:
    raw = [probs_dict.get(c, 1e-10) for c in candidates]
    total = sum(raw)
    w_arr = np.array([v / total for v in raw], dtype=np.float64)
    cands_enc = _encode_words_numpy(candidates, wl)
    n_patterns = 3 ** wl

    cands_set = set(candidates)
    pool = list(candidates)
    remaining = max_pool - len(pool)
    for w in vocab:
        if w not in cands_set and remaining > 0:
            pool.append(w)
            remaining -= 1

    best_H, best_g = -1.0, candidates[0]
    best_is_cand = best_g in cands_set
    pool_enc = _encode_words_numpy(pool, wl)
    for i, g in enumerate(pool):
        fbs = _feedbacks_numpy(pool_enc[i], cands_enc, wl)
        H = _entropy(fbs, w_arr, n_patterns)
        g_is_cand = g in cands_set
        if H > best_H + 1e-9 or (abs(H - best_H) <= 1e-9 and g_is_cand and not best_is_cand):
            best_H, best_g, best_is_cand = H, g, g_is_cand
    return best_g


# ─── Probes y expected-cost (se usan en freq y buster) ──────────────────────

def _f_hat(n: int) -> float:
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 2.0
    return max(1.0, math.log2(n) * 0.5 + 0.8)


def _expected_score(guess: str, candidates: Sequence[str], weights: dict[str, float], wl: int) -> float:
    win_pat = tuple([2] * wl)
    part = defaultdict(list)
    for w in candidates:
        part[framework_feedback(w, guess)].append(w)
    total_w = sum(weights.get(w, 0.0) for w in candidates) or 1.0
    score = 0.0
    for pat, group in part.items():
        p_f = sum(weights.get(w, 0.0) for w in group) / total_w
        score += p_f * (1.0 if pat == win_pat else 1.0 + _f_hat(len(group)))
    return score


def _gen_probe_nonwords(candidates: Sequence[str], wl: int, n: int = 20) -> list[str]:
    ambiguous = set()
    pos_sets = [set() for _ in range(wl)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_sets[i].add(ch)
    for s in pos_sets:
        if len(s) > 1:
            ambiguous.update(s)
    letter_freq = defaultdict(int)
    for w in candidates:
        for ch in set(w):
            letter_freq[ch] += 1
    for ch, _ in sorted(letter_freq.items(), key=lambda x: -x[1]):
        ambiguous.add(ch)
        if len(ambiguous) >= wl + 4:
            break
    pool = list(ambiguous)[: wl + 4]
    cand_set = set(candidates)
    non_words, seen = [], set()
    if len(pool) < wl:
        pool.extend(list(SPANISH_LETTERS[: wl - len(pool)]))
    for combo in itertools.permutations(pool, wl):
        if len(non_words) >= n:
            break
        nw = ''.join(combo)
        if nw in cand_set or nw in seen:
            continue
        seen.add(nw)
        non_words.append(nw)
    return non_words[:n]


def _dynamic_best(candidates: Sequence[str], vocab: Sequence[str], wl: int,
                  probs_dict: dict[str, float], max_pool: int = 400, n_probes: int = 20) -> str:
    weights = _normalize_weights(candidates, probs_dict)
    if len(candidates) <= 3:
        pool = list(candidates)
    else:
        vocab_sample = vocab[:max_pool] if len(vocab) > max_pool else list(vocab)
        probes = _gen_probe_nonwords(candidates, wl, n=n_probes)
        pool = list(set(candidates) | set(vocab_sample) | set(probes))
    best, best_s = candidates[0], float('inf')
    cand_set = set(candidates)
    for g in pool:
        s = _expected_score(g, candidates, weights, wl)
        if s < best_s - 1e-9 or (abs(s - best_s) <= 1e-9 and g in cand_set and best not in cand_set):
            best, best_s = g, s
    return best


# ─── Cluster detection/buster (solo en T3 fallback, n ≤ 8) ──────────────────

def _detect_cluster(candidates: Sequence[str], wl: int) -> tuple[int, tuple[int, ...], dict[int, set[str]]]:
    if not candidates:
        return 0, (), {}
    pos_sets = [set() for _ in range(wl)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_sets[i].add(ch)
    var_pos = [i for i, s in enumerate(pos_sets) if len(s) > 1]
    if len(var_pos) == 1:
        return 1, tuple(var_pos), {var_pos[0]: pos_sets[var_pos[0]]}
    if len(var_pos) == 2:
        return 2, tuple(var_pos), {var_pos[0]: pos_sets[var_pos[0]], var_pos[1]: pos_sets[var_pos[1]]}
    return 0, (), {}


def _build_cluster_buster(var_pos: tuple[int, ...], var_letters: dict[int, set[str]], wl: int) -> str | None:
    letters = []
    for p in var_pos:
        letters.extend(list(var_letters[p]))
    if not letters:
        return None
    # simple spread: rotate variable letters off their original positions
    letters = letters[:wl] if len(letters) >= wl else (letters + list(SPANISH_LETTERS))[:wl]
    # place them in different positions to avoid greens
    buster = ['a'] * wl
    for i, ch in enumerate(letters):
        target = (i + 1) % wl
        buster[target] = ch
    return ''.join(buster)


def _cluster_buster_best(candidates: Sequence[str], wl: int, probs: dict[str, float]) -> tuple[str | None, float]:
    kind, var_pos, var_letters = _detect_cluster(candidates, wl)
    if kind == 0 or len(candidates) > 8:
        return None, float('inf')
    buster = _build_cluster_buster(var_pos, var_letters, wl)
    if not buster:
        return None, float('inf')
    weights = _normalize_weights(candidates, probs)
    score = _expected_score(buster, candidates, weights, wl)
    return buster, score


# ─── Expected cost directo ───────────────────────────────────────────────────

def _expected_cost_direct(candidates: Sequence[str], probs_dict: dict[str, float], turns_left: int) -> float:
    n = len(candidates)
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0
    raw = [probs_dict.get(w, 1e-10) for w in candidates]
    total = sum(raw)
    p_best = max(raw) / total
    n_remaining = n - 1
    if n_remaining == 1:
        e_remaining = 1.0
    elif turns_left <= 1:
        e_remaining = float("inf")
    elif n_remaining == 2:
        e_remaining = 1.5
    else:
        e_remaining = math.log2(n_remaining) + 1.0
    return p_best * 1.0 + (1.0 - p_best) * (1.0 + e_remaining)


# ─── Estrategias por turno ──────────────────────────────────────────────────

def _choose_t4(candidates: Sequence[str], vocab: Sequence[str], wl: int, mode: str,
               probs: dict[str, float]) -> str:
    n = len(candidates)
    if n <= 2:
        return _most_probable(candidates, probs)
    if n <= T4_SAFE_GUESS_MAX_CANDS:
        safe_g, found = _find_safe_guess(candidates, vocab, 2, probs)
        if found:
            if mode == "frequency":
                e_direct = _expected_cost_direct(candidates, probs, turns_left=3)
                p_best = _best_prob(candidates, probs)
                e_safe = (1.0 - p_best) * 2.0 + p_best * 1.0 if safe_g in candidates else 2.0
                if e_direct < e_safe - 1e-9:
                    return _most_probable(candidates, probs)
            return safe_g
    if mode == "frequency":
        p_best = _best_prob(candidates, probs)
        if p_best > T4_DIRECT_PROB_THRESHOLD_FREQ:
            return _most_probable(candidates, probs)
        return _dynamic_best(candidates, vocab, wl, probs)
    p_best = _best_prob(candidates, probs)
    if p_best > T4_DIRECT_PROB_THRESHOLD_UNI:
        return _most_probable(candidates, probs)
    return _best_entropy_guess_vocab(candidates, vocab, wl, probs, max_pool=300)


def _choose_t5(candidates: Sequence[str], vocab: Sequence[str], wl: int, mode: str,
               probs: dict[str, float]) -> str:
    n = len(candidates)
    if n <= 2:
        return _most_probable(candidates, probs)
    if n == 3:
        safe_g, found = _find_safe_guess(candidates, vocab, 1, probs)
        if found:
            return safe_g
        return _most_probable(candidates, probs)
    if mode == "frequency":
        return _dynamic_best(candidates, vocab, wl, probs)
    return _best_entropy_guess_vocab(candidates, vocab, wl, probs, max_pool=300)


def _choose_t3_runtime(candidates: Sequence[str], vocab: Sequence[str], wl: int, mode: str,
                       probs: dict[str, float]) -> str:
    n = len(candidates)
    if n <= 1:
        return candidates[0] if candidates else vocab[0]
    if n == 2:
        return _most_probable(candidates, probs)
    # small cluster buster attempt
    buster, b_score = _cluster_buster_best(candidates, wl, probs)
    # main paths
    if mode == "frequency":
        p_best = _best_prob(candidates, probs)
        if p_best > T3_DIRECT_PROB_THRESHOLD_FREQ:
            return _most_probable(candidates, probs)
        dyn = _dynamic_best(candidates, vocab, wl, probs)
        weights = _normalize_weights(candidates, probs)
        dyn_score = _expected_score(dyn, candidates, weights, wl)
        if buster and b_score + 1e-9 < dyn_score:
            return buster
        return dyn
    p_best = _best_prob(candidates, probs)
    if p_best > T3_DIRECT_PROB_THRESHOLD_UNI:
        return _most_probable(candidates, probs)
    ent = _best_entropy_guess_vocab(candidates, vocab, wl, probs, max_pool=400)
    if buster:
        weights = _normalize_weights(candidates, probs)
        ent_score = _expected_score(ent, candidates, weights, wl)
        if b_score + 1e-9 < ent_score:
            return buster
    return ent


# ─── Clase principal ─────────────────────────────────────────────────────────

class RG5_gabriel_regina(Strategy):
    _t2_tables: dict[str, dict] = {}
    _t3_tables: dict[str, dict] = {}
    _tables_loaded: set[str] = set()

    @property
    def name(self) -> str:
        return "RG5_gabriel_regina"

    @classmethod
    def _load_tables(cls, wl: int, mode: str) -> None:
        key = f"{wl}_{mode}"
        if key in cls._tables_loaded:
            return
        t2_path = _REPO_ROOT / f"t2_table_{wl}_{mode}.json"
        t3_path = _REPO_ROOT / f"t3_table_{wl}_{mode}.json"
        cls._t2_tables[key] = json.load(open(t2_path, encoding="utf-8")) if t2_path.exists() else {}
        cls._t3_tables[key] = json.load(open(t3_path, encoding="utf-8")) if t3_path.exists() else {}
        cls._tables_loaded.add(key)

    def begin_game(self, config: GameConfig) -> None:
        self._wl = config.word_length
        self._mode = config.mode
        self._vocab = list(config.vocabulary)
        self._probs = dict(config.probabilities)
        self._opener = OPENERS[self._wl][self._mode]
        self._max_g = config.max_guesses
        RG5_gabriel_regina._load_tables(self._wl, self._mode)
        self._t2 = RG5_gabriel_regina._t2_tables.get(f"{self._wl}_{self._mode}", {})
        self._t3 = RG5_gabriel_regina._t3_tables.get(f"{self._wl}_{self._mode}", {})

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        turn = len(history) + 1
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)
        if not candidates:
            return self._vocab[0]
        if len(candidates) == 1:
            return candidates[0]

        wl, mode, vocab, probs = self._wl, self._mode, self._vocab, self._probs

        if turn == 1:
            return self._opener

        if turn == 2:
            pat1 = "".join(str(x) for x in history[0][1])
            g2 = self._t2.get(pat1)
            if g2:
                return g2
            if mode == "frequency":
                return _dynamic_best(candidates, vocab, wl, probs)
            return _best_entropy_guess_vocab(candidates, vocab, wl, probs, max_pool=500)

        if turn == 3:
            pat1 = "".join(str(x) for x in history[0][1])
            pat2 = "".join(str(x) for x in history[1][1])
            state = f"{pat1}|{pat2}"
            entry = self._t3.get(state)
            if entry:
                g = entry.get("guess") if isinstance(entry, dict) else entry
                if g:
                    return g
            return _choose_t3_runtime(candidates, vocab, wl, mode, probs)

        if turn == 4:
            return _choose_t4(candidates, vocab, wl, mode, probs)

        if turn == 5:
            return _choose_t5(candidates, vocab, wl, mode, probs)

        return _most_probable(candidates, probs)

