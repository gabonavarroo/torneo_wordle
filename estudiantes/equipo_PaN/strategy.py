# estudiantes/equipo_PaN/strategy.py
"""
Estrategia híbrida Wordle (4/5/6, uniform/frequency).

FIX CRÍTICO:
- No usamos bitmasks (1<<idx). Con 'ñ' (y cualquier caracter fuera de a-z) explota.
- Usamos alfabeto dinámico por juego y precomputamos uniq_count por palabra.

Requisitos:
1) Levenshtein con DP + memoria O(L) y cache dict.
2) Estructuras centrales en diccionarios: self.cache, self.word_info, self.stats.
3) Levenshtein SOLO como ranking dentro de candidatos válidos (ya filtrados por Wordle).
4) self.exploit + regla concreta (T[L], p_star[L]) + time guard fallback.
"""

from __future__ import annotations

import math
import time
import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import filter_candidates


# ------------------------- Levenshtein DP O(L) -------------------------
def levenshtein_dp_ol(a: str, b: str, cache: dict[tuple[str, str], int]) -> int:
    if a == b:
        return 0
    key = (a, b) if a <= b else (b, a)
    if key in cache:
        return cache[key]

    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cb = b[j - 1]
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            v = ins if ins < dele else dele
            if sub < v:
                v = sub
            cur[j] = v
        prev = cur

    d = prev[lb]
    cache[key] = d
    return d


def pattern_code(secret_codes: tuple[int, ...], guess_codes: tuple[int, ...]) -> int:
    """Feedback Wordle (0/1/2) -> entero base-3. Maneja duplicados."""
    L = len(secret_codes)
    pat = [0] * L

    remaining: dict[int, int] = {}
    for s in secret_codes:
        remaining[s] = remaining.get(s, 0) + 1

    for i in range(L):
        g = guess_codes[i]
        s = secret_codes[i]
        if g == s:
            pat[i] = 2
            remaining[g] -= 1

    for i in range(L):
        if pat[i] == 2:
            continue
        g = guess_codes[i]
        cnt = remaining.get(g, 0)
        if cnt > 0:
            pat[i] = 1
            remaining[g] = cnt - 1

    code = 0
    for x in pat:
        code = code * 3 + x
    return code


def safe_argmax(x: np.ndarray) -> int:
    return int(np.argmax(x)) if x.size else 0


# ------------------------------ Estrategia ------------------------------
class MiEstrategia(Strategy):
    @property
    def name(self) -> str:
        return "MiEstrategia_equipo_PaN"

    def begin_game(self, config: GameConfig) -> None:
        # Dicts obligatorios
        self.cache: dict = {}
        self.word_info: dict[str, dict] = {}
        self.stats: dict = {}

        self.exploit = False
        self._L = int(config.word_length)
        self._mode = str(config.mode)
        self._allow_non_words = bool(config.allow_non_words)

        # time guard
        self._t0 = time.perf_counter()
        self.stats["hard_time_limit"] = 4.85

        vocab = list(config.vocabulary)
        n = len(vocab)

        # Probabilidades
        probs = np.empty(n, dtype=np.float64)
        for i, w in enumerate(vocab):
            probs[i] = float(config.probabilities.get(w, 0.0))
        s = float(probs.sum())
        probs[:] = probs / s if s > 0 else (1.0 / max(1, n))

        # Alfabeto dinámico (incluye 'ñ' y cualquier otro caracter presente)
        alphabet = sorted({c for w in vocab for c in w})
        A = len(alphabet)
        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        idx_to_char = alphabet  # lista indexable

        # Codes y uniq_count (sin bitmask)
        L = self._L
        codes = np.empty((n, L), dtype=np.uint16)  # uint16 por si A>255
        uniq_count = np.empty(n, dtype=np.uint8)
        for i, w in enumerate(vocab):
            row = [char_to_idx[c] for c in w]
            codes[i, :] = row
            uniq_count[i] = len(set(row))

        self.stats["data"] = {
            "words": vocab,
            "probs": probs,
            "codes": codes,
            "uniq": uniq_count,
            "word_to_idx": {w: i for i, w in enumerate(vocab)},
            "alphabet": alphabet,
            "A": A,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
        }

        self.cache["lev"] = {}  # memo Levenshtein
        self.cache["cand_state"] = {
            "hist_len": 0,
            "candidates": vocab,
            "cand_idx": np.arange(n, dtype=np.int32),
        }

        # Parámetros exploit + performance (mismos valores base para 4/5/6,
        # sin hacks específicos para un solo caso).
        self.stats["T"] = {4: 4, 5: 6, 6: 8}
        self.stats["K"] = {4: 220, 5: 280, 6: 320}
        # N_eval[L]: máximo de candidatos que usamos para el cálculo de
        # expected information (si hay más, se submuestrean dentro).
        self.stats["N_eval"] = {4: 900, 5: 650, 6: 550}
        # K_eval[L]: tamaño del pool refinado para expected remaining.
        self.stats["K_eval"] = {4: 60, 5: 55, 6: 50}
        self.stats["p_star"] = {
            "uniform": {4: 0.60, 5: 0.60, 6: 0.60},
            "frequency": {4: 0.18, 5: 0.12, 6: 0.10},
        }

        # Pesos scoring barato
        self.stats["w_pos"] = 1.00
        self.stats["w_uniq"] = 0.70
        self.stats["w_prob"] = 0.90 if self._mode == "frequency" else 0.25
        self.stats["w_rep_pen"] = 0.35
        self.stats["w_lev"] = 0.40

        # Starter cover guess (posible non-word)
        self.stats["starter_guess"] = self._build_cover_guess(
            cand_idx=self.cache["cand_state"]["cand_idx"], history=[]
        )

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        if (time.perf_counter() - self._t0) > self.stats["hard_time_limit"]:
            return self._fallback_maxprob(history)

        candidates, cand_idx = self._get_candidates(history)
        if not candidates:
            data = self.stats["data"]
            cand_idx = np.arange(len(data["words"]), dtype=np.int32)

        if cand_idx.size == 1:
            return self.stats["data"]["words"][int(cand_idx[0])]

        if self.exploit:
            return self._pick_best_candidate_fast(cand_idx)

        L = self._L
        T = self.stats["T"].get(L, 6)
        p_star = self.stats["p_star"][self._mode][L]

        data = self.stats["data"]
        probs = data["probs"][cand_idx]
        s = float(probs.sum())
        probs_norm = probs / s if s > 0 else probs
        max_p = float(probs_norm.max()) if probs_norm.size else 0.0

        if cand_idx.size <= T or max_p >= p_star:
            self.exploit = True
            return self._pick_best_candidate_fast(cand_idx)

        # primer guess: cover
        if len(history) == 0 and self._allow_non_words:
            g0 = self.stats["starter_guess"]
            if isinstance(g0, str) and len(g0) == L:
                return g0

        pos_freq, let_freq = self._letter_stats(cand_idx)
        proto = self._prototype_from_posfreq(pos_freq)

        top_idx = self._topk_indices_by_prob(cand_idx, self.stats["K"][L])

        # Siempre que quede tiempo suficiente intentamos una pasada de expected
        # information (approx entropy) usando un subconjunto acotado de
        # candidatos. Si no llega o no mejora, caemos al scoring barato.
        if (time.perf_counter() - self._t0) < 4.2:
            pool = self._build_guess_pool(top_idx, pos_freq, let_freq, history, proto)
            gbest = self._best_by_expected_remaining(cand_idx, pool)
            if gbest is not None:
                return gbest

        # Fallback estable y barato
        return self._best_by_cheap_score(cand_idx, top_idx, pos_freq, let_freq, history, proto)

    # ---------------- candidates (filter_candidates) ----------------
    def _get_candidates(self, history):
        state = self.cache["cand_state"]
        word_to_idx = self.stats["data"]["word_to_idx"]

        hlen = len(history)
        if hlen == 0:
            state["hist_len"] = 0
            state["candidates"] = self.stats["data"]["words"]
            state["cand_idx"] = np.arange(len(state["candidates"]), dtype=np.int32)
            return state["candidates"], state["cand_idx"]

        if state.get("hist_len", 0) == hlen - 1:
            g, pat = history[-1]
            cand = filter_candidates(state["candidates"], g, pat)
        else:
            cand = self.stats["data"]["words"]
            for g, pat in history:
                cand = filter_candidates(cand, g, pat)

        if not cand:
            state["hist_len"] = hlen
            state["candidates"] = []
            state["cand_idx"] = np.empty(0, dtype=np.int32)
            return [], state["cand_idx"]

        idx = np.array([word_to_idx[w] for w in cand], dtype=np.int32)
        state["hist_len"] = hlen
        state["candidates"] = cand
        state["cand_idx"] = idx
        return cand, idx

    # ---------------- fast pick ----------------
    def _fallback_maxprob(self, history):
        _, cand_idx = self._get_candidates(history)
        if cand_idx.size == 0:
            return self.stats["data"]["words"][0]
        return self._pick_best_candidate_fast(cand_idx)

    def _pick_best_candidate_fast(self, cand_idx: np.ndarray) -> str:
        data = self.stats["data"]
        probs = data["probs"][cand_idx]
        best_local = safe_argmax(probs)
        return data["words"][int(cand_idx[best_local])]

    # ---------------- stats ----------------
    def _letter_stats(self, cand_idx: np.ndarray):
        data = self.stats["data"]
        codes = data["codes"]
        A = int(data["A"])

        probs = data["probs"][cand_idx]
        s = float(probs.sum())
        probs = probs / s if s > 0 else np.full_like(probs, 1.0 / max(1, probs.size), dtype=np.float64)

        sub = codes[cand_idx]  # (m, L)
        L = self._L

        pos_freq = np.zeros((L, A), dtype=np.float64)
        for i in range(L):
            pos_freq[i] = np.bincount(sub[:, i], weights=probs, minlength=A)

        wrep = np.repeat(probs, L)
        let_freq = np.bincount(sub.reshape(-1), weights=wrep, minlength=A)
        return pos_freq, let_freq

    def _prototype_from_posfreq(self, pos_freq: np.ndarray) -> str:
        idx_to_char = self.stats["data"]["idx_to_char"]
        return "".join(idx_to_char[int(np.argmax(pos_freq[i]))] for i in range(self._L))

    # ---------------- cover guess ----------------
    def _build_cover_guess(self, cand_idx: np.ndarray, history) -> str:
        data = self.stats["data"]
        idx_to_char = data["idx_to_char"]

        pos_freq, let_freq = self._letter_stats(cand_idx)
        L = self._L

        used_chars = set()
        for g, _ in history:
            used_chars.update(g)

        order = np.argsort(-let_freq)
        chosen = []
        for li in order:
            c = idx_to_char[int(li)]
            if c in used_chars:
                continue
            chosen.append(int(li))
            if len(chosen) >= L:
                break
        if len(chosen) < L:
            for li in order:
                chosen.append(int(li))
                if len(chosen) >= L:
                    break

        chosen_set = set(chosen)
        out = []
        for i in range(L):
            best = None
            best_val = -1.0
            for li in list(chosen_set):
                v = float(pos_freq[i, li])
                if v > best_val:
                    best_val = v
                    best = li
            if best is None:
                best = chosen[i]
            out.append(idx_to_char[int(best)])
            chosen_set.discard(best)

        guess = "".join(out)
        return guess[:L].ljust(L, idx_to_char[0])

    # ---------------- top-k ----------------
    def _topk_indices_by_prob(self, cand_idx: np.ndarray, k: int) -> np.ndarray:
        probs = self.stats["data"]["probs"][cand_idx]
        m = probs.size
        if m <= k:
            return cand_idx
        part = np.argpartition(-probs, kth=k - 1)[:k]
        top = cand_idx[part]
        order = np.argsort(-self.stats["data"]["probs"][top])
        return top[order]

    # ---------------- cheap scoring + levenshtein ----------------
    def _build_guess_pool(self, top_idx, pos_freq, let_freq, history, proto):
        data = self.stats["data"]
        words, codes, uniq, probs = data["words"], data["codes"], data["uniq"], data["probs"]

        seen = {g for g, _ in history}
        w_pos, w_uniq, w_prob, w_rep, w_lev = (
            self.stats["w_pos"],
            self.stats["w_uniq"],
            self.stats["w_prob"],
            self.stats["w_rep_pen"],
            self.stats["w_lev"],
        )
        lev_cache = self.cache["lev"]

        scored = []
        for idx in top_idx:
            i = int(idx)
            w = words[i]
            if w in seen:
                continue
            row = codes[i]

            s_pos = 0.0
            for p in range(self._L):
                s_pos += float(pos_freq[p, int(row[p])])

            used = set()
            s_u = 0.0
            for p in range(self._L):
                li = int(row[p])
                if li in used:
                    continue
                used.add(li)
                s_u += float(let_freq[li])

            rep_pen = w_rep * (self._L - int(uniq[i]))
            pterm = math.log(float(probs[i]) + 1e-12)

            score = w_pos * s_pos + w_uniq * s_u + w_prob * pterm - rep_pen
            # Levenshtein SOLO para ranking dentro de candidatos
            score -= w_lev * levenshtein_dp_ol(w, proto, lev_cache)
            scored.append((score, w))

        scored.sort(key=lambda t: t[0], reverse=True)
        pool = [w for _, w in scored[: self.stats["K_eval"][self._L]]]

        if self._allow_non_words and len(history) <= 1:
            cover = self._build_cover_guess(top_idx, history)
            if cover not in seen:
                pool.append(cover)

        return pool if pool else [words[int(top_idx[0])]]

    def _best_by_cheap_score(self, cand_idx, top_idx, pos_freq, let_freq, history, proto):
        data = self.stats["data"]
        words, codes, uniq, probs = data["words"], data["codes"], data["uniq"], data["probs"]

        seen = {g for g, _ in history}
        if self._allow_non_words and len(history) <= 1 and cand_idx.size > 250:
            cover = self._build_cover_guess(cand_idx, history)
            if cover not in seen:
                return cover

        w_pos, w_uniq, w_prob, w_rep, w_lev = (
            self.stats["w_pos"],
            self.stats["w_uniq"],
            self.stats["w_prob"],
            self.stats["w_rep_pen"],
            self.stats["w_lev"],
        )
        lev_cache = self.cache["lev"]

        best_w = words[int(top_idx[0])]
        best_s = -1e18

        for idx in top_idx:
            i = int(idx)
            w = words[i]
            if w in seen:
                continue
            row = codes[i]

            s_pos = 0.0
            for p in range(self._L):
                s_pos += float(pos_freq[p, int(row[p])])

            used = set()
            s_u = 0.0
            for p in range(self._L):
                li = int(row[p])
                if li in used:
                    continue
                used.add(li)
                s_u += float(let_freq[li])

            rep_pen = w_rep * (self._L - int(uniq[i]))
            pterm = math.log(float(probs[i]) + 1e-12)

            score = w_pos * s_pos + w_uniq * s_u + w_prob * pterm - rep_pen
            score -= w_lev * levenshtein_dp_ol(w, proto, lev_cache)

            if score > best_s:
                best_s = score
                best_w = w

        return best_w

    # ---------------- expected remaining (pequeño) ----------------
    def _encode_word(self, w: str) -> tuple[int, ...]:
        m = self.stats["data"]["char_to_idx"]
        return tuple(m[c] for c in w)

    def _best_by_expected_remaining(self, cand_idx, guess_pool):
        if (time.perf_counter() - self._t0) > 4.35:
            return None

        data = self.stats["data"]
        probs_all = data["probs"][cand_idx]
        s = float(probs_all.sum())
        probs = (
            probs_all / s
            if s > 0
            else np.full_like(probs_all, 1.0 / max(1, probs_all.size), dtype=np.float64)
        )

        # Para no explotar el tiempo cuando hay muchos candidatos, usamos solo
        # un subconjunto de hasta N_eval[L] candidatos para estimar la
        # información esperada. Se seleccionan por probabilidad (top-N), lo que
        # funciona tanto en uniform como en frequency.
        L = self._L
        N_eval = self.stats["N_eval"].get(L, probs.size)
        m = probs.size
        if m <= N_eval:
            eval_idx = cand_idx
            probs_eval = probs
        else:
            k = N_eval
            part = np.argpartition(-probs, k - 1)[:k]
            eval_idx = cand_idx[part]
            probs_eval = probs[part]

        cand_codes_arr = data["codes"][eval_idx]
        cand_codes = [tuple(int(x) for x in row) for row in cand_codes_arr]

        n_patterns = 3 ** L

        best_guess, best_obj = None, float("inf")
        w2i = data["word_to_idx"]

        # Peso del término de “hit directo”: en frequency nos importa algo más
        # acertar pronto que en uniform, pero sin dominar a la entropía.
        if self._mode == "frequency":
            alpha_hit = 0.25
        else:
            alpha_hit = 0.15

        for g in guess_pool:
            if len(g) != L:
                continue
            if (time.perf_counter() - self._t0) > 4.5:
                break

            gc = self._encode_word(g)
            masses = [0.0] * n_patterns

            for i, sc in enumerate(cand_codes):
                pc = pattern_code(sc, gc)
                masses[pc] += float(probs_eval[i])

            # Entropía aproximada de la partición inducida por g:
            # ent = -sum_k p_k * log(p_k), con p_k masa de probabilidad en cada
            # patrón. Esto funciona en uniform y en frequency.
            ent = 0.0
            for mk in masses:
                if mk > 0.0:
                    ent -= mk * math.log(mk + 1e-12)

            # bonus por hit directo: probabilidad de que g sea el secreto
            p_hit = 0.0
            gi = w2i.get(g)
            if gi is not None:
                for j, idx in enumerate(cand_idx):
                    if int(idx) == int(gi):
                        p_hit = float(probs[j])
                        break

            # Queremos maximizar entropía y también p_hit. Minimizar obj
            # equivale a maximizar ambos términos.
            obj = -ent - alpha_hit * p_hit
            if obj < best_obj:
                best_obj, best_guess = obj, g

        return best_guess
