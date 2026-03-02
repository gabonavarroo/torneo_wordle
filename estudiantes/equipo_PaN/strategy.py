import math
import time
from collections import defaultdict
from strategy import Strategy, GameConfig
from wordle_env import feedback as wordle_feedback, filter_candidates

class MiEstrategia_equipo_PaN(Strategy):
    @property
    def name(self) -> str:
        return "MiEstrategia_equipo_PaN"

    def begin_game(self, config: GameConfig) -> None:
        self._t0 = time.perf_counter()
        self._L = config.word_length
        self._mode = config.mode
        self._weighted = (self._mode == "frequency")
        self._vocab = list(config.vocabulary)
        
        # Probabilidades
        raw_probs = {w: float(config.probabilities.get(w, 1.0)) for w in self._vocab}
        total_p = sum(raw_probs.values())
        self._prob_map = {w: (p / total_p) for w, p in raw_probs.items()}

        # Openings
        openings = {4: "caos", 5: "areos", 6: "aserto"} 
        self._first_guess = openings.get(self._L, self._vocab[0])
        
        # MEMORIA CACHÉ: Guarda los feedbacks calculados para no repetir trabajo
        self._fb_cache = {}

    def _get_pattern_int(self, c: str, g: str) -> int:
        """Versión con memoria para ser ultrarrápida."""
        key = c + g
        if key not in self._fb_cache:
            fb = wordle_feedback(c, g)
            v = 0
            for i, x in enumerate(fb):
                v += x * (3 ** i)
            self._fb_cache[key] = v
        return self._fb_cache[key]

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        if not history:
            return self._first_guess

        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)
        
        n_cand = len(candidates)
        if n_cand <= 2:
            return max(candidates, key=lambda w: self._prob_map.get(w, 0))

        # LÍMITE REDUCIDO: 60 palabras son más que suficientes y súper rápidas
        pool_limit = 60 if self._L < 6 else 40
        
        if n_cand > pool_limit:
            pool = self._get_smart_pool(candidates, limit=pool_limit)
        else:
            pool = self._vocab if n_cand > 2 else candidates

        return self._pick_best_entropy(candidates, pool)

    def _get_smart_pool(self, candidates, limit):
        freqs = defaultdict(int)
        for w in candidates:
            for char in set(w): freqs[char] += 1
        
        def score(w):
            return sum(freqs[c] for c in set(w))

        top_cands = sorted(candidates, key=score, reverse=True)[:limit//2]
        top_vocab = sorted(self._vocab, key=score, reverse=True)[:limit//2]
        return list(set(top_cands + top_vocab))

    def _pick_best_entropy(self, candidates, pool):
        best_guess = candidates[0]
        max_ent = -1.0
        
        cand_probs = [self._prob_map.get(c, 0) for c in candidates]
        sum_p = sum(cand_probs)
        norm_probs = [p / sum_p for p in cand_probs] if sum_p > 0 else [1/len(candidates)]*len(candidates)

        for g in pool:
            # CHEQUEO DE TIEMPO: Si quedan menos de 0.5s, detiene la búsqueda 
            # y usa la mejor palabra que haya encontrado hasta el momento.
            if time.perf_counter() - self._t0 > 4.5:
                break
                
            counts = defaultdict(float)
            for i, c in enumerate(candidates):
                p_idx = self._get_pattern_int(c, g)
                counts[p_idx] += norm_probs[i] if self._weighted else 1.0
            
            ent = 0.0
            div = 1.0 if self._weighted else len(candidates)
            for val in counts.values():
                p = val / div
                ent -= p * math.log2(p)
            
            is_cand = g in candidates
            adj_ent = ent + (1e-4 if is_cand else 0) + (self._prob_map.get(g, 0) * 0.01)

            if adj_ent > max_ent:
                max_ent = adj_ent
                best_guess = g
                
        return best_guess