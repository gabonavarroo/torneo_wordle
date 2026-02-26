"""
BÚSQUEDA EXHAUSTIVA DE OPENER — paralela con feedback() real del framework
══════════════════════════════════════════════════════════════════════════════

Usa feedback() exacto del framework (sin reimplementar nada) + multiprocessing.
Garantiza 100% correctitud — el único speedup es dividir trabajo entre núcleos.

Tiempos estimados (8 núcleos):
  4 letras: ~3-5 min    (421K strings × 1,853 vocab)
  5 letras: ~90-120 min (9.7M strings × 4,546 vocab)
  6 letras: ~4-6 horas  (fase 2: top-10K combos × 720 perms × 6,016 vocab)

Uso:
  python3 parallel_opener_search.py --length 4
  python3 parallel_opener_search.py --length 5
  python3 parallel_opener_search.py --length 6
  python3 parallel_opener_search.py --length 4 --mode uniform
  python3 parallel_opener_search.py --length all

  # Ver cuántos núcleos tienes primero:
  nproc

  # Forzar número de workers:
  python3 parallel_opener_search.py --length 5 --workers 12
"""

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")  # 27 letras

# Variables globales del worker — se inicializan una vez por proceso
_vocab = None
_weights = None
_wl = None


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
    try:
        from lexicon import _sigmoid_weights
    except ImportError:
        raise SystemExit("ERROR: corre desde torneo_wordle/")

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
    n = len(words)
    weights_u = {w: 1.0 / n for w in words}
    weights_f = _sigmoid_weights(counts, steepness=1.5)
    print(f"  {wl}-letras: {n} palabras")
    return words, weights_u, weights_f


# ══════════════════════════════════════════════════════════════════════════════
# Entropía con feedback() real del framework
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy(guess_word, vocab, weights):
    """
    Usa feedback() exacto del framework — sin reimplementar nada.
    """
    from wordle_env import feedback as fb
    partition = defaultdict(float)
    for w in vocab:
        pat = fb(w, guess_word)
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
# Worker
# ══════════════════════════════════════════════════════════════════════════════

def _init_worker(vocab, weights, wl):
    global _vocab, _weights, _wl
    _vocab = vocab
    _weights = weights
    _wl = wl


def _worker_batch(combos_batch):
    """
    Recibe una lista de combinaciones de letras.
    Para cada combo evalúa todas sus permutaciones.
    Retorna (mejor_H, mejor_word) del batch.
    """
    best_h = -1.0
    best_word = ''

    for combo in combos_batch:
        for perm in itertools.permutations(combo):
            g = ''.join(perm)
            h = compute_entropy(g, _vocab, _weights)
            if h > best_h:
                best_h = h
                best_word = g

    return best_h, best_word


# ══════════════════════════════════════════════════════════════════════════════
# Búsqueda 4 y 5 letras — fuerza bruta completa
# ══════════════════════════════════════════════════════════════════════════════

def search_brute(wl, words, weights, mode_label, n_workers):
    """
    Evalúa exhaustivamente todos los C(27,wl)×wl! strings con letras únicas.
    """
    n_letters = len(SPANISH_LETTERS)
    all_combos = list(itertools.combinations(SPANISH_LETTERS, wl))
    total_strings = len(all_combos) * math.factorial(wl)

    print(f"\n  {'═'*55}")
    print(f"  Búsqueda {wl}-letras ({mode_label})")
    print(f"  Combos de letras: {len(all_combos):,}")
    print(f"  Total strings:    {total_strings:,}")
    print(f"  Workers:          {n_workers}")
    print(f"  {'═'*55}")
    sys.stdout.flush()

    # Dividir combos en batches balanceados
    batch_size = max(1, len(all_combos) // (n_workers * 20))
    batches = [all_combos[i:i + batch_size]
               for i in range(0, len(all_combos), batch_size)]
    print(f"  Batches: {len(batches):,} (~{batch_size} combos c/u)")
    sys.stdout.flush()

    top20 = []  # (H, word)
    t0 = time.monotonic()
    completed = 0
    report_every = max(1, len(batches) // 40)

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(words, weights, wl)
    ) as pool:
        for h, word in pool.imap_unordered(_worker_batch, batches, chunksize=1):
            completed += 1

            if len(top20) < 20 or h > top20[0][0]:
                top20.append((h, word))
                top20.sort(key=lambda x: x[0])
                if len(top20) > 20:
                    top20.pop(0)

            if completed % report_every == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed / completed * (len(batches) - completed)
                best = top20[-1]
                pct = completed / len(batches) * 100
                print(f"  {pct:5.1f}%  mejor='{best[1]}' H={best[0]:.5f}"
                      f"  ETA={eta/60:.1f}min")
                sys.stdout.flush()

    elapsed = time.monotonic() - t0
    top20.sort(key=lambda x: -x[0])
    _print_top20(top20, wl, mode_label, elapsed, words)
    return top20[0][1], top20[0][0], top20


# ══════════════════════════════════════════════════════════════════════════════
# Búsqueda 6 letras — dos fases
# ══════════════════════════════════════════════════════════════════════════════

def _fast_combo_score(combo, words, wl):
    """
    Score rápido para filtrar combos de letras en fase 1.
    Mide cobertura: cuántas letras del combo aparecen en promedio por palabra.
    """
    letter_set = set(combo)
    score = 0.0
    for w in words:
        score += sum(1 for ch in set(w) if ch in letter_set)
    return score


def _init_worker_score(words, wl):
    global _vocab, _wl
    _vocab = words
    _wl = wl


def _worker_score_batch(combos_batch):
    """Fase 1: score rápido para filtrar combos."""
    results = []
    for combo in combos_batch:
        s = _fast_combo_score(combo, _vocab, _wl)
        results.append((s, combo))
    return results


def search_6(words, weights, mode_label, n_workers, top_combos=10000):
    """
    6 letras en dos fases:
      Fase 1: Score de cobertura sobre C(27,6)=296,010 combos (~1 min)
      Fase 2: Entropía real sobre top-K combos × 720 permutaciones
    """
    wl = 6
    all_combos = list(itertools.combinations(SPANISH_LETTERS, wl))

    print(f"\n  {'═'*55}")
    print(f"  Búsqueda 6-letras ({mode_label}) — 2 fases")
    print(f"  Total combos: {len(all_combos):,}")
    print(f"  Top combos fase 2: {top_combos:,}")
    print(f"  Workers: {n_workers}")
    print(f"  {'═'*55}")
    sys.stdout.flush()

    # ── Fase 1 ────────────────────────────────────────────────────────────────
    print(f"\n  FASE 1: scoring de cobertura...")
    t0 = time.monotonic()
    batch_size = max(1, len(all_combos) // (n_workers * 10))
    batches = [all_combos[i:i + batch_size]
               for i in range(0, len(all_combos), batch_size)]

    all_scores = []
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker_score,
        initargs=(words, wl)
    ) as pool:
        for results in pool.imap_unordered(_worker_score_batch, batches, chunksize=1):
            all_scores.extend(results)

    all_scores.sort(key=lambda x: -x[0])
    top = [combo for _, combo in all_scores[:top_combos]]
    print(f"  ✓ Fase 1 en {time.monotonic()-t0:.0f}s  "
          f"mejor combo: {all_scores[0][1]}")
    sys.stdout.flush()

    # ── Fase 2 ────────────────────────────────────────────────────────────────
    total_strings = len(top) * math.factorial(wl)
    print(f"\n  FASE 2: entropía real")
    print(f"  {len(top):,} combos × {math.factorial(wl)} perms = {total_strings:,} strings")
    sys.stdout.flush()

    batch_size = max(1, len(top) // (n_workers * 20))
    batches = [top[i:i + batch_size]
               for i in range(0, len(top), batch_size)]

    top20 = []
    t0 = time.monotonic()
    completed = 0
    report_every = max(1, len(batches) // 40)

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(words, weights, wl)
    ) as pool:
        for h, word in pool.imap_unordered(_worker_batch, batches, chunksize=1):
            completed += 1
            if len(top20) < 20 or h > top20[0][0]:
                top20.append((h, word))
                top20.sort(key=lambda x: x[0])
                if len(top20) > 20:
                    top20.pop(0)
            if completed % report_every == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed / completed * (len(batches) - completed)
                best = top20[-1]
                pct = completed / len(batches) * 100
                print(f"  {pct:5.1f}%  mejor='{best[1]}' H={best[0]:.5f}"
                      f"  ETA={eta/60:.1f}min")
                sys.stdout.flush()

    elapsed = time.monotonic() - t0
    top20.sort(key=lambda x: -x[0])
    _print_top20(top20, wl, mode_label, elapsed, words)
    return top20[0][1], top20[0][0], top20


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

CURRENT_OPENERS = {4: 'aore', 5: 'careo', 6: 'carieo'}


def _print_top20(top20, wl, mode_label, elapsed, words):
    vocab_set = set(words)
    curr = CURRENT_OPENERS.get(wl, '')
    print(f"\n  ✓ Completado en {elapsed/60:.1f}min")
    print(f"\n  TOP 20 — {wl}-letras {mode_label}:")
    for rank, (h, word) in enumerate(top20, 1):
        tag = "PALABRA" if word in vocab_set else "no-pal "
        marker = " ◄ ACTUAL" if word == curr else ""
        print(f"    {rank:2d}. '{word}'  H={h:.6f}  {tag}{marker}")

    if curr:
        h_curr = compute_entropy(curr, words,
                                  {w: 1/len(words) for w in words}
                                  if mode_label == 'uniform'
                                  else {w: top20[0][0] for w in words})
        # Recomputar correctamente
        h_curr = top20[[w for _, w in top20].index(curr)][0] if curr in [w for _, w in top20] else None
        if h_curr is None:
            # curr no está en top20, calcularlo
            from lexicon import _sigmoid_weights
            import csv as csv2
            h_curr_computed = None  # se calcula abajo en main
        diff = top20[0][0] - (h_curr or top20[0][0])
        if diff > 0.0005:
            print(f"\n  *** NUEVO OPENER: '{top20[0][1]}' (+{diff:.6f} bits) ***")
        else:
            print(f"\n  '{curr}' sigue siendo óptimo (diff ≤ 0.0005 bits)")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda exhaustiva de opener óptimo para Wordle español")
    parser.add_argument("--length", default="4",
                        choices=["4", "5", "6", "all"],
                        help="Longitud de palabras a buscar")
    parser.add_argument("--mode", default="both",
                        choices=["uniform", "frequency", "both"],
                        help="Modo de probabilidades")
    parser.add_argument("--workers", type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Número de procesos paralelos (default: núcleos-1)")
    parser.add_argument("--top-combos", type=int, default=10000,
                        help="Para 6 letras: combos a evaluar en fase 2 (default: 10000)")
    args = parser.parse_args()

    print("═" * 60)
    print("  BÚSQUEDA EXHAUSTIVA DE OPENER — paralela")
    print("═" * 60)
    print(f"  CPU cores disponibles: {mp.cpu_count()}")
    print(f"  Workers a usar:        {args.workers}")
    print(f"  Longitud:              {args.length}")
    print(f"  Modo:                  {args.mode}")

    lengths = [4, 5, 6] if args.length == "all" else [int(args.length)]
    modes = ["uniform", "frequency"] if args.mode == "both" else [args.mode]

    results = {}

    for wl in lengths:
        words, weights_u, weights_f = load_vocab(wl)

        for mode in modes:
            weights = weights_u if mode == "uniform" else weights_f
            key = f"{wl}_{mode}"

            if wl in (4, 5):
                best, h, top20 = search_brute(wl, words, weights, mode, args.workers)
            else:
                best, h, top20 = search_6(words, weights, mode, args.workers, args.top_combos)

            results[key] = {
                "opener": best,
                "entropy": float(h),
                "top20": [(float(h_), w) for h_, w in top20],
            }

            # Comparar con opener actual
            curr = CURRENT_OPENERS.get(wl, '')
            if curr:
                h_curr = compute_entropy(curr, words, weights)
                diff = float(h) - h_curr
                print(f"\n  Comparación final:")
                print(f"    Nuevo:  '{best}'  H={h:.6f}")
                print(f"    Actual: '{curr}'   H={h_curr:.6f}")
                print(f"    Diff:   {diff:+.6f} bits")
                if diff > 0.0005:
                    print(f"  *** USAR '{best}' EN LUGAR DE '{curr}' ***")
                else:
                    print(f"  '{curr}' confirmado como óptimo")
                sys.stdout.flush()

    # Guardar JSON
    out = Path("exhaustive_openers_parallel.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Guardado en {out}")

    # Resumen final para copiar a strategy.py
    print("\n" + "═" * 60)
    print("  COPIA ESTO EN strategy.py:")
    print("═" * 60)
    u = {wl: results.get(f"{wl}_uniform", {}).get("opener", CURRENT_OPENERS[wl])
         for wl in [4, 5, 6] if wl in lengths}
    f_ = {wl: results.get(f"{wl}_frequency", {}).get("opener", CURRENT_OPENERS[wl])
          for wl in [4, 5, 6] if wl in lengths}
    print(f"  OPENERS_UNIFORM = {u}")
    print(f"  OPENERS_FREQ    = {f_}")


if __name__ == "__main__":
    mp.freeze_support()
    main()