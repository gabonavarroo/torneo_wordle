"""
BÚSQUEDA EXHAUSTIVA DE OPENER — versión rápida con NumPy + multiprocessing
══════════════════════════════════════════════════════════════════════════════

Dos optimizaciones combinadas:

1. FEEDBACK VECTORIZADO CON NUMPY
   En vez de llamar feedback(w, guess) 1,853 veces por string en Python puro,
   pre-encodificamos el vocabulario como matriz de enteros y computamos todos
   los feedbacks en una sola operación NumPy.
   Speedup: ~50-100x

2. MULTIPROCESSING
   Dividimos los 421K strings entre todos los núcleos disponibles.
   Speedup: ~N× núcleos

   Combinado (8 núcleos × 80x NumPy): ~640x total
   4 letras: 18 min → ~2 segundos
   5 letras: 12 horas → ~1 minuto

Uso:
  python3 fast_opener_search.py --length 4
  python3 fast_opener_search.py --length 5
  python3 fast_opener_search.py --length 6
  python3 fast_opener_search.py --length 4 --workers 8   # forzar N workers
  python3 fast_opener_search.py --length 5 --mode frequency
"""

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from lexicon import _sigmoid_weights
    print("✓ lexicon importado")
except ImportError:
    raise SystemExit("ERROR: corre desde torneo_wordle/")


SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")  # 27 letras
CHAR_TO_IDX = {ch: i for i, ch in enumerate(SPANISH_LETTERS)}


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
    weights_u = np.ones(len(words)) / len(words)
    weights_f = _sigmoid_weights(counts, steepness=1.5)
    weights_f_arr = np.array([weights_f[w] for w in words])
    weights_f_arr /= weights_f_arr.sum()
    print(f"  {wl}-letras: {len(words)} palabras")
    return words, weights_u, weights_f_arr


# ══════════════════════════════════════════════════════════════════════════════
# Feedback vectorizado con NumPy
# ══════════════════════════════════════════════════════════════════════════════

def encode_words(words: list[str], wl: int) -> np.ndarray:
    """
    Convierte lista de palabras a matriz de enteros (N × wl).
    Cada letra se mapea a su índice en SPANISH_LETTERS (0-26).
    """
    n = len(words)
    mat = np.zeros((n, wl), dtype=np.int8)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            mat[i, j] = CHAR_TO_IDX.get(ch, 0)
    return mat


def compute_feedback_batch(guess_enc: np.ndarray,
                            secrets_enc: np.ndarray,
                            wl: int) -> np.ndarray:
    """
    Calcula el feedback de `guess` contra TODOS los secretos en una operación.

    guess_enc:   array de forma (wl,) — guess codificado
    secrets_enc: matriz de forma (N, wl) — todos los secretos codificados

    Retorna: array de forma (N,) con el feedback codificado como entero base-3
             Ejemplo: (2,1,0,2,1) → 2×3⁴ + 1×3³ + 0×3² + 2×3¹ + 1×3⁰ = 222

    Algoritmo replica exactamente el algoritmo de dos pasadas del framework:
      1. Asignar verdes (2) donde guess[i] == secret[i]
      2. Para los no-verdes, asignar amarillos (1) si la letra está en el
         secreto en otra posición (respetando conteos)
    """
    N = secrets_enc.shape[0]
    result = np.zeros(N, dtype=np.int32)

    # Paso 1: verdes
    greens = (secrets_enc == guess_enc[np.newaxis, :])  # (N, wl) bool

    # Paso 2: amarillos — para cada posición no-verde, verificar si la letra
    # del guess aparece en el secreto en una posición no-verde no ya contada
    # Esto requiere manejar conteos de letras → hacemos un loop por posición
    # pero vectorizado sobre los N secretos

    yellows = np.zeros((N, wl), dtype=bool)

    for i in range(wl):
        if greens[:, i].all():
            continue  # todos son verde en esta posición → no amarillo

        guess_ch = guess_enc[i]
        not_green_i = ~greens[:, i]  # (N,) — secretos donde pos i no es verde

        # Para cada secreto, contar cuántas veces aparece guess_ch en
        # posiciones no-verdes del secreto que no hayan sido asignadas ya
        # (incluyendo posiciones anteriores del guess)
        secret_ch_positions = (secrets_enc == guess_ch)  # (N, wl)
        # Quitar posiciones verdes del secreto
        available = secret_ch_positions & ~greens  # (N, wl)

        # Cuántas veces ya "consumimos" guess_ch en posiciones anteriores
        # (ya sea verde en pos <i con esa letra, o amarillo en pos <i)
        consumed = np.zeros(N, dtype=np.int32)
        for j in range(i):
            if guess_enc[j] == guess_ch:
                consumed += (greens[:, j] | yellows[:, j]).astype(np.int32)

        # Disponibles en el secreto
        available_count = available.sum(axis=1)  # (N,)

        # Es amarillo si: pos i no es verde, Y hay disponibles no consumidos
        yellows[:, i] = not_green_i & (available_count > consumed)

    # Codificar resultado: verde=2, amarillo=1, gris=0
    pattern_mat = np.zeros((N, wl), dtype=np.int32)
    pattern_mat[greens] = 2
    pattern_mat[yellows] = 1

    # Convertir a entero base-3 para usar como clave de partición
    powers = np.array([3**j for j in range(wl - 1, -1, -1)], dtype=np.int32)
    result = (pattern_mat * powers[np.newaxis, :]).sum(axis=1)

    return result


def compute_entropy_vectorized(guess_enc: np.ndarray,
                                secrets_enc: np.ndarray,
                                weights: np.ndarray,
                                wl: int) -> float:
    """
    Calcula la entropía ponderada de un guess usando NumPy.
    Mucho más rápido que el loop Python puro.
    """
    feedbacks = compute_feedback_batch(guess_enc, secrets_enc, wl)

    # Sumar pesos por patrón
    n_patterns = 3 ** wl
    pattern_weights = np.bincount(feedbacks, weights=weights,
                                   minlength=n_patterns)

    # Entropía
    mask = pattern_weights > 0
    p = pattern_weights[mask]
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


# ══════════════════════════════════════════════════════════════════════════════
# Worker para multiprocessing
# ══════════════════════════════════════════════════════════════════════════════

# Variables globales del worker (inicializadas una vez por proceso)
_worker_secrets_enc = None
_worker_weights = None
_worker_wl = None


def _init_worker(secrets_enc, weights, wl):
    """Inicializar estado del worker (se llama una vez por proceso)."""
    global _worker_secrets_enc, _worker_weights, _worker_wl
    _worker_secrets_enc = secrets_enc
    _worker_weights = weights
    _worker_wl = wl


def _worker_batch(combos_batch: list[tuple]) -> tuple[float, str]:
    """
    Procesa un batch de combinaciones de letras.
    Para cada combo evalúa todas sus permutaciones y retorna el mejor (H, word).
    """
    best_h = -1.0
    best_word = ''
    wl = _worker_wl

    for combo in combos_batch:
        for perm in itertools.permutations(combo):
            g = ''.join(perm)
            # Codificar el guess
            guess_enc = np.array([CHAR_TO_IDX.get(ch, 0) for ch in g],
                                   dtype=np.int8)
            h = compute_entropy_vectorized(guess_enc, _worker_secrets_enc,
                                            _worker_weights, wl)
            if h > best_h:
                best_h = h
                best_word = g

    return best_h, best_word


# ══════════════════════════════════════════════════════════════════════════════
# Búsqueda principal
# ══════════════════════════════════════════════════════════════════════════════

def search_fast(wl: int, words: list[str], weights: np.ndarray,
                mode_label: str, n_workers: int) -> tuple[str, float, list]:
    """
    Búsqueda exhaustiva paralela sobre todos los strings con letras únicas.
    """
    n_letters = len(SPANISH_LETTERS)
    n_combos = math.comb(n_letters, wl)
    total_strings = n_combos * math.factorial(wl)

    print(f"\n  Búsqueda {wl}-letras ({mode_label})")
    print(f"  Combinaciones de letras: {n_combos:,}")
    print(f"  Total strings: {total_strings:,}")
    print(f"  Workers: {n_workers}")
    print(f"  Vocab: {len(words)}")
    sys.stdout.flush()

    # Pre-encodificar vocabulario (se comparte entre workers via fork)
    secrets_enc = encode_words(words, wl)

    # Dividir combos en batches para los workers
    all_combos = list(itertools.combinations(SPANISH_LETTERS, wl))

    # Batch size: ~1000 combos por batch para buen balance de carga
    batch_size = max(1, min(1000, n_combos // (n_workers * 10)))
    batches = [all_combos[i:i + batch_size]
               for i in range(0, len(all_combos), batch_size)]

    print(f"  Batches: {len(batches):,} (tamaño ~{batch_size})")
    print(f"  Iniciando...")
    sys.stdout.flush()

    t0 = time.monotonic()
    top20 = []  # lista de (H, word)

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(secrets_enc, weights, wl)
    ) as pool:
        completed = 0
        for h, word in pool.imap_unordered(_worker_batch, batches,
                                            chunksize=1):
            completed += 1

            if len(top20) < 20 or h > top20[0][0]:
                top20.append((h, word))
                top20.sort(key=lambda x: x[0])
                if len(top20) > 20:
                    top20.pop(0)

            if completed % max(1, len(batches) // 50) == 0:
                elapsed = time.monotonic() - t0
                pct = completed / len(batches) * 100
                eta = elapsed / completed * (len(batches) - completed)
                best = top20[-1]
                print(f"    {pct:5.1f}% [{completed:,}/{len(batches):,}] "
                      f"mejor='{best[1]}'(H={best[0]:.4f}) "
                      f"~{eta:.0f}s restantes")
                sys.stdout.flush()

    elapsed = time.monotonic() - t0
    top20.sort(key=lambda x: -x[0])

    # Imprimir resultados
    vocab_set = set(words)
    current = {4: 'aore', 5: 'careo', 6: 'carieo'}
    curr = current.get(wl, '')

    print(f"\n  ✓ Completado en {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"\n  TOP 20 — {wl}-letras {mode_label}:")
    for rank, (h, word) in enumerate(top20, 1):
        tag = "PALABRA" if word in vocab_set else "no-pal "
        marker = " ◄ ACTUAL" if word == curr else ""
        print(f"    {rank:2d}. '{word}'  H={h:.6f}  {tag}{marker}")

    # Comparar con opener actual
    if curr:
        curr_enc = np.array([CHAR_TO_IDX.get(ch, 0) for ch in curr],
                             dtype=np.int8)
        h_curr = compute_entropy_vectorized(curr_enc, secrets_enc, weights, wl)
        diff = top20[0][0] - h_curr
        print(f"\n  Opener actual '{curr}': H={h_curr:.6f}")
        if diff > 0.0005:
            print(f"  *** MEJORA REAL: '{top20[0][1]}' es mejor "
                  f"(+{diff:.6f} bits) ***")
        else:
            print(f"  '{curr}' es óptimo (diff={diff:+.6f} bits, "
                  f"< umbral 0.0005)")

    sys.stdout.flush()
    return top20[0][1], top20[0][0], top20


# ══════════════════════════════════════════════════════════════════════════════
# Búsqueda 6 letras con dos fases
# ══════════════════════════════════════════════════════════════════════════════

def search_6_fast(words: list[str], weights: np.ndarray,
                  mode_label: str, n_workers: int,
                  top_combos: int = 8000) -> tuple[str, float, list]:
    """
    6 letras: dos fases con NumPy + multiprocessing.

    Fase 1: Scoring rápido vectorizado sobre C(27,6)=296,010 combos (~10s)
    Fase 2: Entropía real sobre top-K combos × 720 permutaciones (~20-40 min)
    """
    wl = 6
    n_combos = math.comb(len(SPANISH_LETTERS), wl)
    print(f"\n  Búsqueda 6-letras ({mode_label}) — 2 fases")
    secrets_enc = encode_words(words, wl)

    # ── Fase 1: Score rápido ──────────────────────────────────────────────────
    print(f"\n  FASE 1: scoring rápido de {n_combos:,} combos de letras...")
    sys.stdout.flush()

    # Encodificar vocab como matriz de presencia de letras
    # vocab_letters[i, j] = 1 si letra j aparece en palabra i
    n_letters = len(SPANISH_LETTERS)
    vocab_letter_presence = np.zeros((len(words), n_letters), dtype=np.int8)
    for i, w in enumerate(words):
        for ch in w:
            idx = CHAR_TO_IDX.get(ch, -1)
            if idx >= 0:
                vocab_letter_presence[i, idx] = 1

    t0 = time.monotonic()
    combo_scores = []
    for combo in itertools.combinations(SPANISH_LETTERS, wl):
        idxs = [CHAR_TO_IDX[ch] for ch in combo]
        # Score = suma de letras del combo presentes en cada palabra
        coverage = vocab_letter_presence[:, idxs].sum(axis=1)  # (N,)
        score = float((coverage * weights).sum())
        combo_scores.append((score, combo))

    combo_scores.sort(key=lambda x: -x[0])
    top = combo_scores[:top_combos]
    print(f"  ✓ Fase 1 en {time.monotonic()-t0:.1f}s  "
          f"mejor combo: {top[0][1]}")
    sys.stdout.flush()

    # ── Fase 2: Entropía real con multiprocessing ─────────────────────────────
    top_combos_list = [combo for _, combo in top]
    total_strings = len(top_combos_list) * math.factorial(wl)
    print(f"\n  FASE 2: entropía real — {len(top_combos_list):,} combos "
          f"× {math.factorial(wl)} perms = {total_strings:,} strings")
    print(f"  Workers: {n_workers}")
    sys.stdout.flush()

    batch_size = max(1, min(200, len(top_combos_list) // (n_workers * 5)))
    batches = [top_combos_list[i:i + batch_size]
               for i in range(0, len(top_combos_list), batch_size)]

    t0 = time.monotonic()
    top20 = []

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(secrets_enc, weights, wl)
    ) as pool:
        completed = 0
        for h, word in pool.imap_unordered(_worker_batch, batches, chunksize=1):
            completed += 1
            if len(top20) < 20 or h > top20[0][0]:
                top20.append((h, word))
                top20.sort(key=lambda x: x[0])
                if len(top20) > 20:
                    top20.pop(0)
            if completed % max(1, len(batches) // 40) == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed / completed * (len(batches) - completed)
                best = top20[-1]
                pct = completed / len(batches) * 100
                print(f"    {pct:5.1f}% mejor='{best[1]}'(H={best[0]:.4f}) "
                      f"~{eta/60:.1f}min restantes")
                sys.stdout.flush()

    elapsed = time.monotonic() - t0
    top20.sort(key=lambda x: -x[0])

    vocab_set = set(words)
    print(f"\n  ✓ Fase 2 en {elapsed/60:.1f}min")
    print(f"\n  TOP 20 — 6-letras {mode_label}:")
    for rank, (h, word) in enumerate(top20, 1):
        tag = "PALABRA" if word in vocab_set else "no-pal "
        marker = " ◄ ACTUAL" if word == 'carieo' else ""
        print(f"    {rank:2d}. '{word}'  H={h:.6f}  {tag}{marker}")

    curr_enc = np.array([CHAR_TO_IDX.get(ch, 0) for ch in 'carieo'],
                         dtype=np.int8)
    h_curr = compute_entropy_vectorized(curr_enc, secrets_enc, weights, wl)
    diff = top20[0][0] - h_curr
    print(f"\n  'carieo' actual: H={h_curr:.6f}  diff={diff:+.6f}")
    if diff > 0.0005:
        print(f"  *** MEJORA: '{top20[0][1]}' (+{diff:.6f} bits) ***")

    return top20[0][1], top20[0][0], top20


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", default="4",
                        choices=["4", "5", "6", "all"])
    parser.add_argument("--mode", default="both",
                        choices=["uniform", "frequency", "both"])
    parser.add_argument("--workers", type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Número de workers (default: núcleos - 1)")
    parser.add_argument("--top-combos", type=int, default=8000,
                        help="Para 6 letras: top combos en fase 2 (default: 8000)")
    args = parser.parse_args()

    print("═" * 65)
    print("  BÚSQUEDA EXHAUSTIVA DE OPENER — NumPy + Multiprocessing")
    print("═" * 65)
    print(f"  CPU cores disponibles: {mp.cpu_count()}")
    print(f"  Workers: {args.workers}")

    lengths = [4, 5, 6] if args.length == "all" else [int(args.length)]
    modes = (["uniform", "frequency"] if args.mode == "both"
             else [args.mode])

    import json
    results = {}

    for wl in lengths:
        words, weights_u, weights_f = load_vocab(wl)

        for mode in modes:
            weights = weights_u if mode == "uniform" else weights_f
            key = f"{wl}_{mode}"

            if wl in (4, 5):
                best, h, top20 = search_fast(
                    wl, words, weights, mode, args.workers)
            else:
                best, h, top20 = search_6_fast(
                    words, weights, mode, args.workers, args.top_combos)

            results[key] = {"opener": best, "entropy": float(h),
                            "top20": [(float(h_), w) for h_, w in top20]}

    # Guardar y resumir
    out = Path("exhaustive_openers_fast.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Guardado en {out}")

    print("\n" + "═" * 65)
    print("  RESUMEN FINAL")
    print("═" * 65)
    u = {wl: results.get(f"{wl}_uniform", {}).get("opener", "?")
         for wl in lengths}
    f_ = {wl: results.get(f"{wl}_frequency", {}).get("opener", "?")
          for wl in lengths}
    print(f"  OPENERS_UNIFORM = {u}")
    print(f"  OPENERS_FREQ    = {f_}")
    print("\n  Si hay cambios respecto a los actuales, re-corre:")
    print("  python3 optimal_dp_4letter.py --mode both")


if __name__ == "__main__":
    mp.freeze_support()
    main()