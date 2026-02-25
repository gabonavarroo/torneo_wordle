"""
diagnostico.py — Perfilador de tiempo para InformationStrategy

Instrumenta cada fase del juego para identificar exactamente dónde
se va el tiempo. Corre con:

    python3 diagnostico.py --length 5 --mode uniform --games 3
    python3 diagnostico.py --length 6 --mode uniform --games 3
    python3 diagnostico.py --length 4 --mode uniform --games 3

La salida te dirá:
  - Cuánto tarda begin_game (opener)
  - Cuánto tarda cada llamada a guess()
  - Cuántos candidatos quedan en cada turno
  - Tamaño del guess_pool en cada turno
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

# Asegurarnos de que el repo raíz esté en el path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from lexicon import load_lexicon
from wordle_env import WordleEnv, feedback, filter_candidates
from strategy import GameConfig


# ── Versión instrumentada de la estrategia ────────────────────────────────────
# Importamos la estrategia real y le añadimos timing sin modificarla.

import importlib.util
strat_path = repo_root / "estudiantes" / "gabriel_regina" / "strategy.py"
spec = importlib.util.spec_from_file_location("student_strategy", strat_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
InformationStrategy = mod.InformationStrategy


class TimedStrategy:
    """Wrapper que mide el tiempo de cada llamada sin tocar la lógica interna."""

    def __init__(self):
        self._inner = InformationStrategy()
        self.timings = []   # Lista de dicts con info por turno

    def begin_game(self, config: GameConfig) -> None:
        t0 = time.perf_counter()
        self._inner.begin_game(config)
        elapsed = time.perf_counter() - t0

        # Info interna disponible tras begin_game
        opener = getattr(self._inner, '_opener', '???')
        vocab_size = len(getattr(self._inner, '_vocab', []))
        non_words = len(getattr(self._inner, '_non_words', []))

        self.timings = []
        self._begin_time = elapsed
        self._opener = opener
        self._vocab_size = vocab_size
        self._non_words_count = non_words

        print(f"\n  [begin_game] {elapsed:.3f}s")
        print(f"    vocab_size   = {vocab_size}")
        print(f"    non_words    = {non_words}")
        print(f"    opener       = '{opener}'  "
              f"({'PALABRA' if opener in getattr(self._inner, '_vocab_set', set()) else 'NO-PALABRA'})")

    def guess(self, history: list) -> str:
        turn = len(history) + 1

        # Reconstruir candidatos para medir su tamaño
        candidates = getattr(self._inner, '_vocab', [])
        for g, pat in history:
            candidates = filter_candidates(candidates, g, pat)

        t0 = time.perf_counter()
        result = self._inner.guess(history)
        elapsed = time.perf_counter() - t0

        print(f"  [guess t{turn}] {elapsed:.3f}s  "
              f"candidatos={len(candidates)}  "
              f"→ '{result}'  "
              f"({'CANDIDATO' if result in set(candidates) else 'NO-PALABRA/FUERA'})")

        self.timings.append({
            'turn': turn,
            'time': elapsed,
            'candidates': len(candidates),
            'guess': result,
        })

        return result


# ── Runner principal ──────────────────────────────────────────────────────────

def run_diagnostic(word_length: int, mode: str, num_games: int, seed: int = 42):
    import random

    print(f"\n{'═'*60}")
    print(f"  DIAGNÓSTICO: {word_length} letras | modo: {mode}")
    print(f"{'═'*60}")

    # Cargar vocabulario
    data_dir = repo_root / "data"
    csv_path = data_dir / f"spanish_{word_length}letter.csv"
    mini_path = data_dir / f"mini_spanish_{word_length}.txt"

    if csv_path.exists():
        lex = load_lexicon(path=str(csv_path), word_length=word_length, mode=mode)
    elif mini_path.exists():
        lex = load_lexicon(path=str(mini_path), word_length=word_length, mode=mode)
    else:
        print(f"  ERROR: No hay datos para {word_length} letras. Corre: python3 download_words.py --all-lengths")
        return

    vocab = lex.words
    probs = dict(lex.probs)

    print(f"  Vocabulario: {len(vocab)} palabras")

    config = GameConfig(
        word_length=word_length,
        vocabulary=tuple(vocab),
        mode=mode,
        probabilities=probs,
        max_guesses=6,
        allow_non_words=True,
    )

    rng = random.Random(seed)
    secrets = rng.sample(vocab, min(num_games, len(vocab)))

    env = WordleEnv(vocabulary=vocab, word_length=word_length, max_guesses=6, allow_non_words=True)

    all_begin_times = []
    all_guess_times = []
    all_total_times = []
    all_guesses_counts = []
    all_solved = []

    for i, secret in enumerate(secrets, 1):
        print(f"\n--- Juego {i}/{len(secrets)} | Secreto: '{secret}' ---")

        strat = TimedStrategy()

        t_game_start = time.perf_counter()
        strat.begin_game(config)
        all_begin_times.append(strat._begin_time)

        env.reset(secret=secret)
        game_guess_times = []

        while not env.game_over():
            guess_word = strat.guess(env.history)
            pat = env.guess(guess_word)
            if strat.timings:
                game_guess_times.append(strat.timings[-1]['time'])

        t_game_total = time.perf_counter() - t_game_start

        solved = env.is_solved()
        n_guesses = len(env.history)
        all_solved.append(solved)
        all_guesses_counts.append(n_guesses)
        all_guess_times.extend(game_guess_times)
        all_total_times.append(t_game_total)

        status = "✓ RESUELTO" if solved else "✗ FALLIDO"
        print(f"  → {status} en {n_guesses} guesses | tiempo total: {t_game_total:.3f}s")
        if t_game_total > 4.0:
            print(f"  ⚠️  RIESGO DE TIMEOUT (>4s)")

    # ── Resumen estadístico ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  RESUMEN ({word_length} letras | {mode})")
    print(f"{'─'*60}")
    print(f"  Juegos: {len(secrets)}  |  Resueltos: {sum(all_solved)}  "
          f"({100*sum(all_solved)//len(secrets)}%)")
    print(f"  Guesses promedio: {sum(all_guesses_counts)/len(all_guesses_counts):.2f}")
    print()
    print(f"  Tiempo begin_game:")
    print(f"    min={min(all_begin_times):.3f}s  "
          f"max={max(all_begin_times):.3f}s  "
          f"prom={sum(all_begin_times)/len(all_begin_times):.3f}s")
    if all_guess_times:
        print(f"  Tiempo por guess():")
        print(f"    min={min(all_guess_times):.4f}s  "
              f"max={max(all_guess_times):.3f}s  "
              f"prom={sum(all_guess_times)/len(all_guess_times):.4f}s")
    print(f"  Tiempo total por juego:")
    print(f"    min={min(all_total_times):.3f}s  "
          f"max={max(all_total_times):.3f}s  "
          f"prom={sum(all_total_times)/len(all_total_times):.3f}s")
    over_limit = sum(1 for t in all_total_times if t > 5.0)
    if over_limit:
        print(f"  ⚠️  {over_limit}/{len(secrets)} juegos superaron 5s (timeout en torneo)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnóstico de timing para InformationStrategy")
    parser.add_argument("--length", type=int, default=5, help="Longitud de palabra (4, 5 o 6)")
    parser.add_argument("--mode", choices=["uniform", "frequency"], default="uniform")
    parser.add_argument("--games", type=int, default=3, help="Juegos a correr (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_diagnostic(args.length, args.mode, args.games, args.seed)