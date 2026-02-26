"""
BÚSQUEDA EXHAUSTIVA DE OPENER — 4, 5 y 6 letras
═════════════════════════════════════════════════

Estrategia por longitud:
  4 letras: fuerza bruta completa sobre strings con letras únicas
            C(27,4)×4! = 421,200 strings × 1,853 = 780M ops (~30 min)

  5 letras: fuerza bruta completa sobre strings con letras únicas
            C(27,5)×5! = 9,687,600 strings × 4,546 = 44B ops (~12 horas)
            → Corre en segunda máquina overnight

  6 letras: fuerza bruta exacta es inviable (213M × 6,016 = 1.28T ops).
            Estrategia en dos fases:
              FASE 1: Score rápido por "cobertura letra×posición" para filtrar
                      el espacio de C(27,6) = 296,010 combinaciones de letras
                      a las top-5,000 (tarda ~1 min)
              FASE 2: Para cada combinación top, evaluar todas sus 6! = 720
                      permutaciones con entropía real (tarda ~2-3 horas)
            Esto garantiza explorar el subespacio más prometedor.

Uso:
  python3 exhaustive_opener_search_all.py --length 4   # ~30 min
  python3 exhaustive_opener_search_all.py --length 5   # ~12 horas
  python3 exhaustive_opener_search_all.py --length 6   # ~2-3 horas
  python3 exhaustive_opener_search_all.py --length all # todo secuencial

  # En segunda máquina (correr en paralelo con 6):
  python3 exhaustive_opener_search_all.py --length 5 --mode uniform
  python3 exhaustive_opener_search_all.py --length 5 --mode frequency

Output: exhaustive_openers.json con los mejores openers por longitud y modo.
"""

import argparse
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

# Letras válidas en español (27)
SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")


# ══════════════════════════════════════════════════════════════════════════════
# Carga de vocabulario
# ══════════════════════════════════════════════════════════════════════════════

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


def load_vocab(wl):
    csv_path = Path(f"data/spanish_{wl}letter.csv")
    pattern = re.compile(rf"^[a-zñ]{{{wl}}}$")
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
    print(f"  {wl}-letras: {len(words)} palabras")
    return words, weights_u, weights_f


# ══════════════════════════════════════════════════════════════════════════════
# Función de entropía
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Score rápido para fase de filtrado (solo para 6 letras)
# ══════════════════════════════════════════════════════════════════════════════

def fast_letter_set_score(letter_combo, vocab, wl):
    """
    Score rápido O(|vocab|) sin calcular feedback completo.
    Mide cuántas palabras del vocab contienen al menos k de las letras del combo.
    Un combo que cubre más palabras en más posiciones tiene mayor entropía potencial.

    Esto es solo para filtrar los C(27,6)=296,010 combos de letras a un top-5000
    antes de evaluar las permutaciones con entropía real.
    """
    letter_set = set(letter_combo)
    score = 0.0
    for w in vocab:
        covered = sum(1 for ch in w if ch in letter_set)
        # Palabras con más letras del combo = más información potencial
        score += covered / wl
    return score


# ══════════════════════════════════════════════════════════════════════════════
# Búsquedas por longitud
# ══════════════════════════════════════════════════════════════════════════════

def search_4_or_5(wl, vocab, weights, mode_label):
    """
    Fuerza bruta completa sobre todos los strings con letras únicas.
    4 letras: ~30 min | 5 letras: ~12 horas
    """
    n_letters = len(SPANISH_LETTERS)
    total = math.comb(n_letters, wl) * math.factorial(wl)

    # Estimación de tiempo basada en medición empírica:
    # feedback() en Python puro ≈ 1-2µs por llamada
    secs_estimate = total * len(vocab) * 1.5e-6
    print(f"\n  Búsqueda exhaustiva {wl}-letras ({mode_label})")
    print(f"  Strings con letras únicas: {total:,}")
    print(f"  Operaciones: {total * len(vocab) / 1e9:.2f}B feedback calls")
    print(f"  Tiempo estimado: ~{secs_estimate/60:.0f} min")
    print(f"  (progreso cada 10,000 strings)...")
    sys.stdout.flush()

    top_k = []
    t0 = time.monotonic()
    checked = 0

    for combo in itertools.combinations(SPANISH_LETTERS, wl):
        for perm in itertools.permutations(combo):
            g = ''.join(perm)
            h = compute_entropy(g, vocab, weights)

            if len(top_k) < 20 or h > top_k[0][0]:
                top_k.append((h, g))
                top_k.sort(key=lambda x: x[0])
                if len(top_k) > 20:
                    top_k.pop(0)

            checked += 1
            if checked % 10_000 == 0:
                elapsed = time.monotonic() - t0
                rate = checked / elapsed
                eta = (total - checked) / rate
                best = top_k[-1]
                print(f"    [{checked:,}/{total:,}] "
                      f"mejor='{best[1]}'(H={best[0]:.4f}) "
                      f"~{eta/60:.1f}min restantes")
                sys.stdout.flush()

    elapsed = time.monotonic() - t0
    top_k.sort(key=lambda x: -x[0])
    _print_results(top_k, wl, mode_label, elapsed, vocab)
    return top_k[0][1], top_k[0][0], top_k


def search_6_two_phase(vocab, weights, mode_label,
                       top_combos=5000, perms_per_combo=None):
    """
    Búsqueda en dos fases para 6 letras (fuerza bruta inviable).

    FASE 1: Filtrar C(27,6)=296,010 combos de letras usando score rápido
            → seleccionar top-5,000 combos más prometedores (~1 min)

    FASE 2: Para cada combo top, evaluar todas sus 720 permutaciones
            con entropía real (~5,000 × 720 × 6,016 = 21.7B ops → ~4-5h)

    Si --fast: solo evaluar la mejor permutación por combo (heurística posicional)
               ~5,000 × 6,016 = 30M ops → ~30 min, menos preciso
    """
    wl = 6
    n_combos = math.comb(len(SPANISH_LETTERS), wl)
    print(f"\n  Búsqueda 2-fases 6-letras ({mode_label})")
    print(f"  Total combos de letras: {n_combos:,}")
    print(f"  Top combos a evaluar exhaustivamente: {top_combos:,}")

    # ── FASE 1: Score rápido para filtrar ────────────────────────────────────
    print(f"\n  FASE 1: Scoring rápido de {n_combos:,} combos...")
    t0 = time.monotonic()
    combo_scores = []

    for i, combo in enumerate(itertools.combinations(SPANISH_LETTERS, wl)):
        s = fast_letter_set_score(combo, vocab, wl)
        combo_scores.append((s, combo))
        if (i + 1) % 50_000 == 0:
            elapsed = time.monotonic() - t0
            eta = (n_combos - i - 1) / (i + 1) * elapsed
            print(f"    [{i+1:,}/{n_combos:,}] ~{eta:.0f}s restantes")

    combo_scores.sort(key=lambda x: -x[0])
    top = combo_scores[:top_combos]
    elapsed = time.monotonic() - t0
    print(f"  ✓ Fase 1 en {elapsed:.0f}s — top combo: {top[0][1]}")

    # ── FASE 2: Entropía real sobre permutaciones ─────────────────────────────
    # Frecuencia letra×posición para elegir permutación inicial inteligente
    pos_freq = [defaultdict(float) for _ in range(wl)]
    for w in vocab:
        for i, ch in enumerate(w):
            pos_freq[i][ch] += 1.0 / len(vocab)

    print(f"\n  FASE 2: Entropía real sobre top-{top_combos:,} combos × 720 permutaciones...")
    total_evals = top_combos * math.factorial(wl)
    print(f"  Total evaluaciones: {total_evals:,} ({total_evals * len(vocab) / 1e9:.1f}B ops)")
    print(f"  Tiempo estimado: ~{total_evals * len(vocab) / 800e6 / 3600:.1f} horas")

    top_k = []
    t0 = time.monotonic()
    checked = 0
    vocab_set = set(vocab)

    for rank, (_, combo) in enumerate(top):
        for perm in itertools.permutations(combo):
            g = ''.join(perm)
            h = compute_entropy(g, vocab, weights)

            if len(top_k) < 20 or h > top_k[0][0]:
                top_k.append((h, g))
                top_k.sort(key=lambda x: x[0])
                if len(top_k) > 20:
                    top_k.pop(0)

        checked += 1
        if checked % 500 == 0:
            elapsed = time.monotonic() - t0
            rate = checked / elapsed
            eta = (top_combos - checked) / rate
            best = top_k[-1]
            print(f"    [{checked:,}/{top_combos:,}] "
                  f"mejor='{best[1]}'(H={best[0]:.4f}) "
                  f"~{eta/60:.1f}min restantes")

    elapsed = time.monotonic() - t0
    top_k.sort(key=lambda x: -x[0])
    _print_results(top_k, wl, mode_label, elapsed, vocab)
    return top_k[0][1], top_k[0][0], top_k


def _print_results(top_k, wl, mode_label, elapsed, vocab):
    vocab_set = set(vocab)
    print(f"\n  ✓ Completado en {elapsed/60:.1f} min")
    print(f"\n  TOP 20 openers {wl}-letras ({mode_label}):")

    # Cargar openers actuales para comparar
    current = {'aore': 4, 'careo': 5, 'carieo': 6}
    curr = current.get(wl, '')

    for rank, (h, word) in enumerate(top_k[:20], 1):
        tag = "✓ PALABRA" if word in vocab_set else "  no-pal "
        marker = " ◄ ACTUAL" if word == curr else ""
        print(f"    {rank:2d}. '{word}'  H={h:.6f}  {tag}{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", default="4",
                        choices=["4", "5", "6", "all"])
    parser.add_argument("--mode", default="both",
                        choices=["uniform", "frequency", "both"])
    parser.add_argument("--top-combos", type=int, default=5000,
                        help="Para 6 letras: cuántos combos de letras evaluar (default: 5000)")
    args = parser.parse_args()

    lengths = [4, 5, 6] if args.length == "all" else [int(args.length)]
    modes_to_run = (["uniform", "frequency"] if args.mode == "both"
                    else [args.mode])

    print("═" * 65)
    print("  BÚSQUEDA EXHAUSTIVA DE OPENER")
    print("═" * 65)

    import json
    results = {}

    for wl in lengths:
        vocab, weights_u, weights_f = load_vocab(wl)

        for mode in modes_to_run:
            weights = weights_u if mode == "uniform" else weights_f
            key = f"{wl}_{mode}"

            if wl in (4, 5):
                best_word, best_h, top20 = search_4_or_5(
                    wl, vocab, weights, mode)
            else:  # wl == 6
                best_word, best_h, top20 = search_6_two_phase(
                    vocab, weights, mode,
                    top_combos=args.top_combos)

            results[key] = {
                "opener": best_word,
                "entropy": best_h,
                "top20": [(h, w) for h, w in top20],
            }

            # Comparar con openers actuales
            current_openers = {4: 'aore', 5: 'careo', 6: 'carieo'}
            curr = current_openers.get(wl, '')
            if curr:
                h_curr = compute_entropy(curr, vocab, weights)
                diff = best_h - h_curr
                if diff > 0.0005:
                    print(f"\n  *** MEJORA: '{best_word}' ({best_h:.6f}) > "
                          f"'{curr}' ({h_curr:.6f}) diff={diff:+.6f} ***")
                else:
                    print(f"\n  '{curr}' sigue siendo óptimo "
                          f"(diff={diff:+.6f})")

    # Guardar resultados
    out = Path("exhaustive_openers.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Resultados guardados en {out}")

    # Resumen final
    print("\n" + "═" * 65)
    print("  RESUMEN — OPENERS ÓPTIMOS CONFIRMADOS")
    print("═" * 65)
    print(f"  OPENERS_UNIFORM = {{", end="")
    for wl in sorted(set(int(k.split("_")[0]) for k in results)):
        key = f"{wl}_uniform"
        if key in results:
            print(f"{wl}: '{results[key]['opener']}', ", end="")
    print("}")
    print(f"  OPENERS_FREQ    = {{", end="")
    for wl in sorted(set(int(k.split("_")[0]) for k in results)):
        key = f"{wl}_frequency"
        if key in results:
            print(f"{wl}: '{results[key]['opener']}', ", end="")
    print("}")


if __name__ == "__main__":
    main()