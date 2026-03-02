"""Strategy template — copy this directory to estudiantes/<your_team>/

Rename the class and the ``name`` property, then implement your logic
in the ``guess`` method.
"""

from __future__ import annotations

from strategy import Strategy, GameConfig
from wordle_env import feedback, filter_candidates
from collections import Counter


class MyStrategy(Strategy):
    """Example strategy — replace with your own logic.
    Reducir con cada guess la lista de plabras posibles
    Pensar en este problema como un problema de combinatoria, pensar las palabras como combinaciones de letras, y cada guess nos ayudara de reducir el espacio"""


    @property
    def name(self) -> str:
        
        return "Minimaxer_bernardor" 

    def begin_game(self, config: GameConfig) -> None:
        """Called once at the start of each game.

        Available information in config:
          - config.word_length    (int)  — 4, 5, or 6
          - config.vocabulary     (tuple) — all valid words (secret is from here)
          - config.mode           (str)  — "uniform" or "frequency"
          - config.probabilities  (dict) — word -> probability (sums to 1)
          - config.max_guesses    (int)  — maximum guesses allowed (typically 6)
          - config.allow_non_words (bool) — True = you can guess ANY letter combo
        """
        self._vocab = list(config.vocabulary)
        self._config = config

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        """Return the next guess.

        Parameters
        ----------
        history : list of (guess, feedback) tuples
            Each feedback is a tuple of ints:
              2 = green (correct letter, correct position)
              1 = yellow (correct letter, wrong position)
              0 = gray (letter not in word)

        Returns
        -------
        str
            A lowercase string of length config.word_length.
            Can be ANY letter combination (not restricted to vocabulary)
            for better information discovery.
        """
        # Filter candidates consistent with all feedback so far
        
        candidates = self._vocab
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        if not candidates:
            return self._vocab[0]
       
        if not history:
            return self._guess_by_letter_freq(candidates)


        # Turnos siguientes: elegir el candidato que más reduce el espacio
        return self._best_candidate(candidates)

    def _guess_by_letter_freq(self, candidates):
        wlen = self._config.word_length

        # Frecuencia de cada letra en cada posición
        pos_freq = [Counter() for _ in range(wlen)]
        for word in candidates:
            for i, ch in enumerate(word):
                pos_freq[i][ch] += 1

        def score_word(word: str) -> tuple:
            # Suma de frecuencia posicional de cada letra (sin repetir letra)
            seen = set()
            total = 0
            for i, ch in enumerate(word):
                if ch not in seen:        
                    total += pos_freq[i][ch]
                    seen.add(ch)
            ends_in_s = word[-1] == "s"   # bonus por terminar en 's' No se si esto es una buena idea, pero en español muchas palabras terminan en 's' y puede ayudar a reducir el espacio de candidatos ademas se me hace interesante la idea 
            return (total, ends_in_s)

        return max(candidates, key=score_word)

    def _best_candidate(self, candidates: list[str]) -> str:
        """
        Busca el mejor candidato para minimizar el tamaño esperado del grupo restante después de hacer el guess.
        """
        mode = self._config.mode
        probs = self._config.probabilities

        best = candidates[0]
        best_score = float("inf")

        for g in candidates:
            
            groups: dict[tuple, float] = {}
            for secret in candidates:
                pat = feedback(secret, g)
                w = probs.get(secret, 1e-9) if mode == "frequency" else 1.0
                groups[pat] = groups.get(pat, 0.0) + w

            
            total = sum(groups.values())
            expected_size = sum((v / total) * v for v in groups.values())

            if expected_size < best_score:
                best_score = expected_size
                best = g

        return best

