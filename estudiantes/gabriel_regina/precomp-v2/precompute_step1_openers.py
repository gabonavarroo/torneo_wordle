"""
PASO 1: Computar los 6 openers óptimos.

Corre desde:  cd ~/ia/wordle/torneo_wordle
Comando:      python3 precompute_step1_openers.py

Tiempo estimado:
  4-letras: ~10 min  (1903 × 1853 = 3.5M feedback calls)
  5-letras: ~90 min  (4596 × 4546 = 20.9M)
  6-letras: ~3 horas (6066 × 6016 = 36.5M)

Output: precomputed_openers.txt con las 6 líneas para copiar a strategy.py
"""

import csv
import itertools
import math
import time
from collections import defaultdict
from pathlib import Path

# Importar feedback del framework
try:
    from wordle_env import feedback
    print("✓ wordle_env importado")
except ImportError:
    raise SystemExit("ERROR: corre desde torneo_wordle/")

# Importar la función REAL de pesos sigmoid de lexicon.py
# Esto garantiza que usamos exactamente las mismas probabilidades que el torneo
try:
    from lexicon import _sigmoid_weights
    print("✓ _sigmoid_weights importado de lexicon.py")
except ImportError:
    raise SystemExit("ERROR: no se pudo importar _sigmoid_weights de lexicon.py")


# ══════════════════════════════════════════════════════════════════════════════
# Carga de vocabulario
# ══════════════════════════════════════════════════════════════════════════════

def load_vocab(wl: int) -> tuple[list[str], dict[str, int], dict[str, float]]:
    """
    Lee data/spanish_{wl}letter.csv y devuelve:
      - words:   lista de palabras (mismo orden que el framework)
      - counts:  dict palabra → count crudo
      - probs_freq: dict palabra → probabilidad frequency (sigmoid-weighted)

    Para uniform: pesos = 1/N (todos iguales, no necesita counts)
    Para frequency: pesos = _sigmoid_weights(counts, steepness=1.5)
                    — EXACTAMENTE lo que hace lexicon.py
    """
    import unicodedata, re

    def strip_accents(text):
        result = []
        for ch in text:
            if ch == "ñ":
                result.append("ñ")
            else:
                decomposed = unicodedata.normalize("NFD", ch)
                result.append("".join(c for c in decomposed
                                      if unicodedata.category(c) != "Mn"))
        return "".join(result)

    csv_path = Path(f"data/spanish_{wl}letter.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}")

    pattern = re.compile(rf"^[a-zñ]{{{wl}}}$")
    seen = set()
    words, counts = [], {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = strip_accents(row['word'].strip().lower())
            if not w or w in seen:
                continue
            if not pattern.match(w):
                continue
            c = int(row['count'])
            if c <= 0:
                continue
            seen.add(w)
            words.append(w)
            counts[w] = c

    words.sort()  # lexicon.py ordena alfabéticamente

    # Probabilidades frequency: sigmoid de log-frecuencia, igual que lexicon.py
    probs_freq = _sigmoid_weights(counts, steepness=1.5)

    print(f"  {wl}-letras: {len(words)} palabras")
    print(f"    Muestra palabras:      {words[:5]}")
    print(f"    Probs freq (top 5):   "
          f"{[round(probs_freq[w], 6) for w in words[:5]]}")

    # Comparar con naive para ver la diferencia del sigmoid
    total_counts = sum(counts.values())
    naive_top = [round(counts[w]/total_counts, 6) for w in words[:5]]
    print(f"    Probs naive (top 5):  {naive_top}  ← NO usamos esto para freq")

    return words, counts, probs_freq


# ══════════════════════════════════════════════════════════════════════════════
# Generación de no-palabras
# ══════════════════════════════════════════════════════════════════════════════

def generate_non_words(vocab: list[str], wl: int, n: int = 50) -> list[str]:
    """
    No-palabras informativas: combina las letras más frecuentes
    en sus posiciones óptimas, sin restricciones fonológicas.
    Usamos pesos uniformes para la generación (independiente del modo).
    """
    vocab_set = set(vocab)
    w_unif = 1.0 / len(vocab)

    pos_freq = [defaultdict(float) for _ in range(wl)]
    for w in vocab:
        for i, ch in enumerate(w):
            pos_freq[i][ch] += w_unif

    overall = defaultdict(float)
    for i in range(wl):
        for ch, freq in pos_freq[i].items():
            overall[ch] += freq

    top = [ch for ch, _ in sorted(overall.items(), key=lambda x: -x[1])]
    pool_size = min(wl + 5, len(top))
    cand_letters = top[:pool_size]

    non_words, seen = [], set()
    for combo in itertools.combinations(cand_letters, wl):
        remaining = list(combo)
        assignment = [''] * wl
        for i in range(wl):
            best = max(remaining, key=lambda ch: pos_freq[i].get(ch, 0.0))
            assignment[i] = best
            remaining.remove(best)
        nw = "".join(assignment)
        if nw not in seen and nw not in vocab_set:
            seen.add(nw)
            non_words.append(nw)
        if len(non_words) >= n:
            break

    return non_words


# ══════════════════════════════════════════════════════════════════════════════
# Entropía ponderada
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy(guess_word: str,
                    candidates: list[str],
                    weights: dict[str, float]) -> float:
    """H(g) = -Σ_f p(f|g) · log₂(p(f|g))"""
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


# ══════════════════════════════════════════════════════════════════════════════
# Encontrar opener óptimo
# ══════════════════════════════════════════════════════════════════════════════

def find_best_opener(vocab: list[str],
                     weights: dict[str, float],
                     non_words: list[str],
                     mode_label: str,
                     wl: int) -> tuple[str, float]:
    """
    Búsqueda exhaustiva del opener con máxima entropía.
    Evalúa TODAS las palabras del vocab + no-palabras generadas.
    Sin límite de tiempo — corremos esto offline una sola vez.
    """
    full_pool = vocab + non_words
    print(f"\n  Opener ({mode_label}, {wl}-letras): "
          f"{len(full_pool)} candidatos × {len(vocab)} vocab...")

    best_guess = vocab[0]
    best_h = -1.0
    t0 = time.monotonic()

    for i, g in enumerate(full_pool):
        h = compute_entropy(g, vocab, weights)
        if h > best_h:
            best_h = h
            best_guess = g

        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            remaining = (len(full_pool) - i - 1) / rate
            is_word = "PALABRA" if g in set(vocab) else "NP"
            print(f"    [{i+1}/{len(full_pool)}] "
                  f"mejor='{best_guess}'(H={best_h:.4f}) "
                  f"~{remaining:.0f}s restantes")

    elapsed = time.monotonic() - t0
    tag = "PALABRA" if best_guess in set(vocab) else "NO-PALABRA"
    print(f"  ✓ '{best_guess}' H={best_h:.4f} [{tag}] — {elapsed:.1f}s")
    return best_guess, best_h


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("PASO 1: Openers óptimos para los 6 escenarios")
    print("=" * 65)

    # Verificar que feedback tiene el orden correcto
    # En el framework: feedback(secreto, guess) → tuple
    print("\nVerificando feedback(secreto, guess)...")
    # Si el secreto es 'arbol' y el guess es 'arbol' → todo verde
    r1 = feedback("arbol", "arbol")
    print(f"  feedback('arbol','arbol') = {r1}  "
          f"{'✓' if r1==(2,2,2,2,2) else '✗ PROBLEMA'}")
    # Si el secreto es 'bbbbb' y el guess es 'aaaaa' → todo gris
    r2 = feedback("bbbbb", "aaaaa")
    print(f"  feedback('bbbbb','aaaaa') = {r2}  "
          f"{'✓' if r2==(0,0,0,0,0) else '✗ PROBLEMA'}")

    results_uniform = {}
    results_freq = {}

    for wl in [4, 5, 6]:
        print(f"\n{'='*65}")
        print(f"LONGITUD {wl} LETRAS")
        print(f"{'='*65}")

        vocab, counts, probs_freq = load_vocab(wl)
        non_words = generate_non_words(vocab, wl, n=50)

        # Pesos uniform: exactamente 1/N para cada palabra
        weights_u = {w: 1.0 / len(vocab) for w in vocab}

        # Pesos frequency: sigmoid(1.5 * (log_count - mu)) normalizado
        # — importado directamente de lexicon.py para garantizar exactitud
        weights_f = probs_freq

        opener_u, h_u = find_best_opener(vocab, weights_u, non_words,
                                          "uniform", wl)
        opener_f, h_f = find_best_opener(vocab, weights_f, non_words,
                                          "frequency", wl)

        results_uniform[wl] = opener_u
        results_freq[wl] = opener_f

    # Resultados finales
    print("\n" + "=" * 65)
    print("RESULTADOS — edita OPENERS en precompute_step2_tables.py:")
    print("=" * 65)
    print(f"\nOPENERS_UNIFORM = {results_uniform}")
    print(f"OPENERS_FREQ    = {results_freq}")

    with open("precomputed_openers.txt", "w") as f:
        f.write(f"OPENERS_UNIFORM = {results_uniform}\n")
        f.write(f"OPENERS_FREQ    = {results_freq}\n")
    print("\nGuardado en precomputed_openers.txt")


if __name__ == "__main__":
    main()