"""
BÚSQUEDA EXHAUSTIVA DE OPENER — 4 letras

Evalúa TODOS los strings posibles de 4 letras (no solo vocabulario + heurísticas)
para encontrar el opener con máxima entropía de forma garantizada.

El espacio es: letras válidas en español = a-z + ñ = 27 letras
27^4 = 531,441 combinaciones posibles

Pero con la restricción de letras únicas (máxima información):
C(27, 4) × 4! = 17,550 × 24 = 421,200 strings — manejable.

Sin restricción: 531,441 strings × 1,853 feedback calls = ~985M operaciones.
En Python puro: ~30-60 min. Aceptable para correr una vez.

Con restricción de letras únicas (casi siempre el óptimo tiene letras distintas):
421,200 × 1,853 = ~780M operaciones — similar.

Estrategia: primero correr con letras únicas (más rápido), luego si el top-10
incluye strings con letras repetidas, verificar exhaustivamente.

Corre desde: cd ~/ia/wordle/torneo_wordle
Comando: python3 exhaustive_opener_search.py
"""

import csv
import itertools
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    from wordle_env import feedback
    from lexicon import _sigmoid_weights
    print("✓ framework importado")
except ImportError:
    raise SystemExit("ERROR: corre desde torneo_wordle/")


WORD_LENGTH = 4
# Letras válidas en español (sin acentos — el framework los normaliza)
SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")  # 27 letras


def _strip_accents(text):
    result = []
    for ch in text:
        if ch == "ñ":
            result.append("ñ")
        else:
            decomposed = unicodedata.normalize("NFD", ch)
            result.append("".join(c for c in decomposed
                                  if unicodedata.category(c) != "Mn"))
    return "".join(result)


def load_vocab():
    csv_path = Path("data/spanish_4letter.csv")
    pattern = re.compile(r"^[a-zñ]{4}$")
    seen, words, counts = set(), [], {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = _strip_accents(row['word'].strip().lower())
            if not w or w in seen or not pattern.match(w):
                continue
            c = int(row['count'])
            if c <= 0:
                continue
            seen.add(w)
            words.append(w)
            counts[w] = c
    words.sort()
    weights_u = {w: 1.0 / len(words) for w in words}
    weights_f = _sigmoid_weights(counts, steepness=1.5)
    print(f"  Vocabulario: {len(words)} palabras")
    return words, weights_u, weights_f


def compute_entropy(guess_word, candidates, weights):
    """H(g) = -Σ_f p(f|g) · log₂(p(f|g))"""
    partition = defaultdict(float)
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


def generate_all_unique_letter_strings(wl=4):
    """
    Genera todos los strings de wl letras con letras ÚNICAS.
    C(27,4) × 4! = 421,200 strings.
    
    Los openers óptimos casi siempre tienen letras distintas porque
    letras repetidas desperdician información.
    """
    count = 0
    for combo in itertools.combinations(SPANISH_LETTERS, wl):
        for perm in itertools.permutations(combo):
            yield ''.join(perm)
            count += 1


def search_exhaustive_unique(vocab, weights, mode_label):
    """
    Búsqueda exhaustiva sobre todos los strings de 4 letras únicas.
    """
    print(f"\n  Búsqueda exhaustiva ({mode_label})...")
    print(f"  Espacio: C(27,4)×4! = {math.comb(27,4) * 24:,} strings con letras únicas")
    print(f"  Operaciones estimadas: {math.comb(27,4) * 24 * len(vocab) / 1e6:.0f}M feedback calls")

    top_k = []   # lista de (H, word) — mantener top 20
    t0 = time.monotonic()
    checked = 0

    for g in generate_all_unique_letter_strings(WORD_LENGTH):
        h = compute_entropy(g, vocab, weights)
        
        # Mantener top-20
        if len(top_k) < 20 or h > top_k[0][0]:
            top_k.append((h, g))
            top_k.sort(key=lambda x: x[0])
            if len(top_k) > 20:
                top_k.pop(0)

        checked += 1
        if checked % 50_000 == 0:
            elapsed = time.monotonic() - t0
            total = math.comb(27, WORD_LENGTH) * math.factorial(WORD_LENGTH)
            rate = checked / elapsed
            eta = (total - checked) / rate
            best_so_far = top_k[-1]
            print(f"    [{checked:,}/{total:,}] "
                  f"mejor='{best_so_far[1]}'(H={best_so_far[0]:.4f}) "
                  f"~{eta:.0f}s restantes")

    elapsed = time.monotonic() - t0
    top_k.sort(key=lambda x: -x[0])  # descendente

    print(f"\n  ✓ Completado en {elapsed/60:.1f} min")
    print(f"\n  TOP 20 openers ({mode_label}):")
    vocab_set = set(vocab)
    for rank, (h, word) in enumerate(top_k[:20], 1):
        is_word = "✓ PALABRA" if word in vocab_set else "  no-pal"
        print(f"    {rank:2d}. '{word}'  H={h:.6f}  {is_word}")

    return top_k[0][1], top_k[0][0], top_k


def main():
    print("=" * 65)
    print("BÚSQUEDA EXHAUSTIVA DE OPENER — 4 letras")
    print("=" * 65)
    print(f"  Letras: {SPANISH_LETTERS}")
    print(f"  Total strings únicos: {math.comb(27, WORD_LENGTH) * math.factorial(WORD_LENGTH):,}")

    vocab, weights_u, weights_f = load_vocab()
    vocab_set = set(vocab)

    results = {}

    for mode, weights in [("uniform", weights_u), ("frequency", weights_f)]:
        best_word, best_h, top20 = search_exhaustive_unique(vocab, weights, mode)
        results[mode] = {"opener": best_word, "entropy": best_h, "top20": top20}

        print(f"\n  RESULTADO {mode}: '{best_word}' H={best_h:.6f}")
        
        # Comparar con lo que teníamos
        h_aore = compute_entropy('aore', vocab, weights)
        diff = best_h - h_aore
        print(f"  Comparación con 'aore': H={h_aore:.6f} (diff={diff:+.6f})")
        if diff > 0.001:
            print(f"  *** MEJORA REAL: '{best_word}' es mejor que 'aore' ***")
        else:
            print(f"  'aore' es óptimo (o casi óptimo) para {mode}")

    # Resumen final
    print("\n" + "=" * 65)
    print("RESUMEN — nuevos openers si mejoran los actuales:")
    print("=" * 65)
    for mode, res in results.items():
        h_aore = compute_entropy('aore', vocab, 
                                  weights_u if mode == "uniform" else weights_f)
        diff = res['entropy'] - h_aore
        status = f"MEJOR (+{diff:.4f})" if diff > 0.001 else f"igual (diff={diff:+.4f})"
        print(f"  {mode}: '{res['opener']}' H={res['entropy']:.6f}  — {status} vs 'aore'")

    print("\n  Si hay mejora, actualiza OPENERS en strategy.py y re-corre los scripts de precomputación.")


if __name__ == "__main__":
    main()