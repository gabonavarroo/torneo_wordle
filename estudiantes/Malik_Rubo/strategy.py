"""Estrategia Wordle tipo solver de entropía adaptada al framework del torneo.

Incluye:
- Autodetección del orden de argumentos de feedback (secret-first vs guess-first).
- Adaptación automática de caps/pools a vocab mini vs full.
- Modo frequency:
  - Proxy rápido por masa^2 (bucket metric) para ranking barato.
  - Ganancia de info EXACTA ponderada (solo cuando quedan pocas soluciones).
- Solve-fast con awareness de intentos restantes:
  - Si remaining <= guesses_left: cerrar (adivinar dentro de remaining).
  - Si remaining > guesses_left (caso crítico): buscar DISCRIMINATOR (separar candidatos).
  - Para 4-6: splittear solo si mejora suficiente el bucket esperado.
- Splitter racional en casos "familia" (fallback).
- Opener óptimo precomputado (minimiza bucket esperado al inicio).
"""

from __future__ import annotations

import inspect
import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from strategy import Strategy, GameConfig
from wordle_env import feedback

Pattern = Tuple[int, ...]


# ============================================================
# Helpers
# ============================================================


def _infer_feedback_secret_first(feedback_fn) -> bool:
    """Infere si feedback(secret, guess) o feedback(guess, secret) por nombres de params."""
    try:
        sig = inspect.signature(feedback_fn)
        params = list(sig.parameters.values())
        if len(params) >= 2:
            p0 = (params[0].name or "").lower()
            p1 = (params[1].name or "").lower()
            if any(k in p0 for k in ("secret", "target", "answer", "solution")):
                return True
            if any(k in p1 for k in ("secret", "target", "answer", "solution")):
                return False
    except Exception:
        pass
    return False


def _tune_by_vocab_size(vocab_size: int) -> dict[str, int]:
    """Ajusta límites según el vocabulario elegido (mini vs full)."""
    if vocab_size <= 1500:
        return {
            "MAX_POOL": 450,
            "TOP_PROB": 120,
            "TOP_COV": 150,
            "TOP_LETTERS": 6,
            "MAX_SYNTHETIC": 120,
            "TOPK_RANK": 40,
        }
    elif vocab_size <= 6000:
        return {
            "MAX_POOL": 800,
            "TOP_PROB": 300,
            "TOP_COV": 300,
            "TOP_LETTERS": 7,
            "MAX_SYNTHETIC": 350,
            "TOPK_RANK": 60,
        }
    else:
        return {
            "MAX_POOL": 1100,
            "TOP_PROB": 450,
            "TOP_COV": 450,
            "TOP_LETTERS": 8,
            "MAX_SYNTHETIC": 600,
            "TOPK_RANK": 70,
        }


def _has_useful_probs(p: dict[str, float], vocab: list[str]) -> bool:
    """Decide si probabilities sirve (no vacía, no todo igual, cubre vocab)."""
    if not p:
        return False
    sample = vocab[: min(200, len(vocab))]
    vals = [float(p.get(w, 0.0)) for w in sample]
    if sum(1 for v in vals if v > 0) < max(5, len(vals) // 10):
        return False
    mn, mx = min(vals), max(vals)
    if abs(mx - mn) < 1e-15:
        return False
    return True


# ============================================================
# Solver interno
# ============================================================


class _WordleEntropySolver:
    """Solver de entropía / bucket metric; con info ponderada exacta cuando aplica."""

    def __init__(
        self,
        soluciones: Sequence[str],
        candidatos: Optional[Sequence[str]] = None,
        word_length: int = 5,
        probs: Optional[Dict[str, float]] = None,
        feedback_secret_first: bool = False,
    ) -> None:
        self._L = word_length
        self.soluciones_all = self._limpiar_candidatos(soluciones)
        base_cands = candidatos if candidatos is not None else soluciones
        self.candidatos_all = self._limpiar_candidatos(base_cands)
        self.restantes = list(self.soluciones_all)

        self._fb_int_cache: dict[tuple[str, str], int] = {}
        self._feedback_secret_first = feedback_secret_first

        # Masa normalizada por solución en modo frequency
        if probs is not None:
            mass: Dict[str, float] = {s: float(probs.get(s, 0.0)) for s in self.soluciones_all}
            total = sum(mass.values())
            if total > 0:
                for k in list(mass.keys()):
                    mass[k] /= total
                self._mass: Optional[Dict[str, float]] = mass
            else:
                self._mass = None
        else:
            self._mass = None

    def _limpiar_candidatos(self, cands: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for g in cands:
            w = g.strip().lower()
            if len(w) != self._L or not w.isalpha():
                continue
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def reset(self) -> None:
        self.restantes = list(self.soluciones_all)

    def _pattern_to_int(self, p: Pattern) -> int:
        x = 0
        m = 1
        for d in p:
            x += d * m
            m *= 3
        return x

    def _feedback_int(self, guess: str, target: str) -> int:
        key = (guess, target)
        cached = self._fb_int_cache.get(key)
        if cached is not None:
            return cached

        if self._feedback_secret_first:
            pat = feedback(target, guess)  # (secret, guess)
        else:
            pat = feedback(guess, target)  # (guess, secret)

        val = self._pattern_to_int(pat)
        self._fb_int_cache[key] = val
        return val

    def aplicar_feedback(self, guess: str, patron: Pattern) -> None:
        guess = guess.strip().lower()
        p_int = self._pattern_to_int(patron)
        self.restantes = [sol for sol in self.restantes if self._feedback_int(guess, sol) == p_int]

    def counts_patrones(self, guess: str, soluciones: Sequence[str]) -> List[int]:
        guess = guess.strip().lower()
        n_patrones = 3 ** self._L
        counts = [0] * n_patrones
        for sol in soluciones:
            idx = self._feedback_int(guess, sol)
            counts[idx] += 1
        return counts

    def counts_patrones_pesados(self, guess: str, soluciones: Sequence[str]) -> List[float]:
        guess = guess.strip().lower()
        n_patrones = 3 ** self._L
        counts = [0.0] * n_patrones
        if self._mass is None:
            return counts
        for sol in soluciones:
            idx = self._feedback_int(guess, sol)
            counts[idx] += self._mass.get(sol, 0.0)
        return counts

    def expected_bucket_metric(self, guess: str, sols: Sequence[str]) -> float:
        """Más bajo = mejor. Uniforme: sum(c^2)/n. Frequency: sum(m^2)."""
        if self._mass is not None:
            counts = self.counts_patrones_pesados(guess, sols)
            return sum(m * m for m in counts)

        counts = self.counts_patrones(guess, sols)
        n = max(1, len(sols))
        return sum(c * c for c in counts) / n

    def ganancia_informacion(self, guess: str, soluciones: Optional[Sequence[str]] = None) -> float:
        """Score de info (mayor es mejor). En frequency usa -bucket_metric."""
        sols = self.restantes if soluciones is None else list(soluciones)
        n = len(sols)
        if n <= 1:
            return 0.0

        if self._mass is not None:
            return -self.expected_bucket_metric(guess, sols)

        h_antes = math.log2(n)
        counts = self.counts_patrones(guess, sols)

        acc = 0.0
        for c in counts:
            if c > 1:
                acc += c * math.log2(c)
        h_despues = acc / n

        return h_antes - h_despues

    def info_gain_weighted_exact(self, guess: str, sols: Sequence[str]) -> float:
        """IG exacta ponderada (solo modo frequency)."""
        if self._mass is None:
            return 0.0

        prior_H = 0.0
        for s in sols:
            ps = self._mass.get(s, 0.0)
            if ps > 0.0:
                prior_H -= ps * math.log2(ps)

        n_pat = 3 ** self._L
        mass_pat = [0.0] * n_pat
        plogp_pat = [0.0] * n_pat

        for s in sols:
            ps = self._mass.get(s, 0.0)
            if ps <= 0.0:
                continue
            idx = self._feedback_int(guess, s)
            mass_pat[idx] += ps
            plogp_pat[idx] += ps * math.log2(ps)

        post = 0.0
        for i, P in enumerate(mass_pat):
            if P > 0.0:
                post += (-plogp_pat[i]) + (P * math.log2(P))

        return prior_H - post

    def mejores_intentos(
        self,
        top_k: int = 10,
        candidatos: Optional[Sequence[str]] = None,
        soluciones: Optional[Sequence[str]] = None,
        *,
        mode: str = "info",  # "info" o "bucket"
    ) -> List[tuple[str, float]]:
        sols = self.restantes if soluciones is None else list(soluciones)
        cands = self.candidatos_all if candidatos is None else list(candidatos)
        candidatos_limpios = self._limpiar_candidatos(cands)
        if not candidatos_limpios or not sols:
            return []

        puntuaciones: List[tuple[str, float]] = []
        if mode == "bucket":
            for g in candidatos_limpios:
                puntuaciones.append((g, -self.expected_bucket_metric(g, sols)))
        else:
            for g in candidatos_limpios:
                puntuaciones.append((g, self.ganancia_informacion(g, sols)))

        puntuaciones.sort(key=lambda x: -x[1])
        return puntuaciones[:top_k]


# ============================================================
# Estrategia del torneo
# ============================================================


class MiEstrategia_Malik_Rubo(Strategy):
    """Estrategia: opener óptimo + discriminators + exact IG cuando conviene."""

    @property
    def name(self) -> str:
        return "MiEstrategia_Malik_Rubo"

    def begin_game(self, config: GameConfig) -> None:
        self._config = config
        self._vocab = list(config.vocabulary)
        self._L = config.word_length
        self._tune = _tune_by_vocab_size(len(self._vocab))

        if config.mode == "frequency":
            p = dict(getattr(config, "probabilities", {}))
            if _has_useful_probs(p, self._vocab):
                self._p = p
                self._use_frequency = True
            else:
                self._p = {w: 1.0 for w in self._vocab}
                self._use_frequency = False
        else:
            self._p = {w: 1.0 for w in self._vocab}
            self._use_frequency = False

        feedback_secret_first = _infer_feedback_secret_first(feedback)

        soluciones = self._vocab
        candidates: List[str] = list(soluciones)

        if config.allow_non_words:
            from collections import Counter

            letter_counts = Counter()
            for w in soluciones:
                letter_counts.update(w)

            top_letters = [c for c, _ in letter_counts.most_common(self._tune["TOP_LETTERS"])]

            synthetic: List[str] = []
            for tup in itertools.product(top_letters, repeat=self._L):
                synthetic.append("".join(tup))
                if len(synthetic) >= self._tune["MAX_SYNTHETIC"]:
                    break

            merged: List[str] = []
            seen: set[str] = set()
            for w in itertools.chain(candidates, synthetic):
                if w not in seen:
                    seen.add(w)
                    merged.append(w)
            candidates = merged

        self._solver = _WordleEntropySolver(
            soluciones=soluciones,
            candidatos=candidates,
            word_length=self._L,
            probs=self._p if self._use_frequency else None,
            feedback_secret_first=feedback_secret_first,
        )

        from collections import Counter

        letter_counts = Counter()
        for v in self._vocab:
            letter_counts.update(set(v))
        self._letter_counts = letter_counts

        self._opener = self._compute_best_opener()

    # ----------------------------
    # Guess logic (con awareness de tiros restantes)
    # ----------------------------

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        # opener
        if not history:
            return self._opener

        # reconstruye estado
        self._solver.reset()
        for g, pat in history:
            self._solver.aplicar_feedback(g, pat)

        remaining = self._solver.restantes
        if not remaining:
            return self._vocab[0]

        played = {g for g, _ in history}

        # tiros restantes
        max_guesses = getattr(self._config, "max_guesses", 6)
        guesses_left = max(0, max_guesses - len(history))

        # pool
        guess_pool = self._build_guess_pool(remaining)
        cand_pool = [w for w in guess_pool if w not in played]
        if not cand_pool:
            return self._best_by_probability(remaining)

        # A) Si caben en los tiros restantes, cierra
        if len(remaining) <= guesses_left:
            return self._best_by_probability(remaining)

        # B) Caso crítico: pocas soluciones pero no caben -> DISCRIMINAR
        # Ej: remaining=3, guesses_left=2
        if len(remaining) <= 12 and guesses_left <= 3:
            disc = self._find_discriminator_guess(remaining, cand_pool)
            if disc is not None:
                return disc

        # C) 4-6: splittear solo si mejora suficiente
        if len(remaining) <= 6:
            best_in = self._best_by_probability(remaining)

            ranked_bucket = self._solver.mejores_intentos(
                top_k=min(self._tune["TOPK_RANK"], 40),
                candidatos=cand_pool,
                soluciones=remaining,
                mode="bucket",
            )
            if ranked_bucket:
                best_bucket_word = ranked_bucket[0][0]
                m_close = self._solver.expected_bucket_metric(best_in, remaining)
                m_split = self._solver.expected_bucket_metric(best_bucket_word, remaining)

                IMPROVEMENT = 0.12
                if (m_close - m_split) >= IMPROVEMENT:
                    return best_bucket_word

            return best_in

        # D) frequency exact IG cuando quedan pocas (baja promedio)
        if getattr(self, "_use_frequency", False) and len(remaining) <= 200:
            prelim = self._solver.mejores_intentos(
                top_k=80,
                candidatos=cand_pool,
                soluciones=remaining,
                mode="bucket",
            )
            best_g, best_s = None, float("-inf")
            for g, _ in prelim:
                s = self._solver.info_gain_weighted_exact(g, remaining)
                if g in remaining:
                    s += 0.02
                if s > best_s:
                    best_s, best_g = s, g
            if best_g is not None:
                return best_g

        # E) fallback: info + tie-break
        mejores = self._solver.mejores_intentos(
            top_k=self._tune["TOPK_RANK"],
            candidatos=cand_pool,
            soluciones=remaining,
            mode="info",
        )
        if not mejores:
            return self._best_by_probability(remaining)

        best_word, best_score = None, float("-inf")
        for g, info in mejores:
            s = self._score_guess(g, info, remaining)
            if s > best_score:
                best_score, best_word = s, g

        return best_word if best_word is not None else self._best_by_probability(remaining)

    # ----------------------------
    # Discriminator helper
    # ----------------------------

    def _find_discriminator_guess(
        self,
        sols: Sequence[str],
        pool: Sequence[str],
    ) -> Optional[str]:
        """Busca un guess que produzca patrones todos distintos para cada sol en sols."""
        sols_list = list(sols)
        if len(sols_list) <= 1:
            return sols_list[0] if sols_list else None

        best = None
        best_distinct = -1

        for g in pool:
            pats = [self._solver._feedback_int(g, s) for s in sols_list]
            distinct = len(set(pats))

            # perfecto: separa a todos
            if distinct == len(sols_list):
                return g

            if distinct > best_distinct:
                best_distinct = distinct
                best = g

        return best

    # ----------------------------
    # Opener
    # ----------------------------

    def _compute_best_opener(self) -> str:
        pool: List[str] = []
        seen: set[str] = set()

        def add(ws: Sequence[str]) -> None:
            for w in ws:
                if w not in seen:
                    seen.add(w)
                    pool.append(w)

        add(self._vocab[: min(len(self._vocab), 800)])

        if getattr(self, "_use_frequency", False):
            add(sorted(self._vocab, key=lambda w: self._p.get(w, 0.0), reverse=True)[:600])

        add(sorted(self._vocab, key=self._coverage_score, reverse=True)[:600])

        pool = pool[:800]

        ranked = self._solver.mejores_intentos(
            top_k=1,
            candidatos=pool,
            soluciones=self._vocab,
            mode="bucket",
        )
        return ranked[0][0] if ranked else self._vocab[0]

    # ----------------------------
    # Base utilities
    # ----------------------------

    def _best_by_probability(self, words: Sequence[str]) -> str:
        ws = list(words)
        if not ws:
            return self._vocab[0]
        if getattr(self, "_use_frequency", False):
            return max(ws, key=lambda w: self._p.get(w, 0.0))
        return ws[0]

    def _coverage_score(self, w: str) -> int:
        return sum(self._letter_counts[c] for c in set(w))

    def _build_guess_pool(self, sols: Sequence[str]) -> List[str]:
        pool: List[str] = []
        seen: set[str] = set()

        def add_many(ws: Sequence[str]) -> None:
            for w in ws:
                if w not in seen:
                    seen.add(w)
                    pool.append(w)

        add_many(list(sols))

        if getattr(self, "_use_frequency", False):
            top_prob = sorted(self._vocab, key=lambda w: self._p.get(w, 0.0), reverse=True)[
                : self._tune["TOP_PROB"]
            ]
            add_many(top_prob)

        top_cov = sorted(self._vocab, key=self._coverage_score, reverse=True)[
            : self._tune["TOP_COV"]
        ]
        add_many(top_cov)

        return pool[: self._tune["MAX_POOL"]]

    def _score_guess(self, g: str, info: float, sols: Sequence[str]) -> float:
        score = info
        if g in sols:
            score += 0.05
        if len(sols) > 50:
            score -= 0.02 * (self._L - len(set(g)))
        if getattr(self, "_use_frequency", False) and len(sols) <= 12:
            score += 0.05 * self._p.get(g, 0.0)
        return score