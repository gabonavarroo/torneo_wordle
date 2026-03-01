"""
precompute_t3.py
────────────────
Precomputa la tabla de T3 para los puzzles de 4, 5 y 6 letras en español.

Para cada estado alcanzable tras T1+T2 con más de 2 candidatos,
encuentra el guess que maximiza la entropía ponderada sobre el
subvocabulario restante.

REGLAS POSICIONALES ESTRICTAS (implementan lo que el juego sabe pero
los números puros no ven automáticamente):
  - Letras grises de T1 y T2 → excluidas del pool de guesses T3
  - Letras amarillas en posición P de T1 o T2 → el guess T3 no puede
    poner esa misma letra en la misma posición P
  - Letras verdes ya confirmadas → no "gastar" esa posición repitiendo
    la misma letra (da 0 info nueva sobre esa posición)

CRITERIO DE SELECCIÓN:
  - Principal: máxima entropía ponderada
  - Desempate (diferencia < 1e-9 bits): más letras nuevas (no usadas
    en T1 ni T2). Si hay empate en letras nuevas también, menor orden
    lexicográfico.

EJECUCIÓN:
  python3 precompute_t3.py --length 4 --workers 8
  python3 precompute_t3.py --length 5 --workers 24
  python3 precompute_t3.py --length 6 --workers 24
  python3 precompute_t3.py --length all --workers 24

SALIDA:
  t3_table_{wl}_uniform.json
  t3_table_{wl}_frequency.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from itertools import permutations
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
DATA_DIR  = REPO_ROOT / "data"

# ── Openers confirmados por búsqueda exhaustiva ───────────────────────────────
OPENERS = {
    4: {"uniform": "aore", "frequency": "aore"},
    5: {"uniform": "careo", "frequency": "careo"}, #PSEUDO CORRECTO, el bueno es sareo, pero careo es casi tan bueno y es el que se usó en T2
    6: {"uniform": "ceriao", "frequency": "ceriao"},
}

# ── T2 tables precomputadas (leídas desde JSON) ───────────────────────────────
T2_TABLES: dict[tuple[int, str], dict[str, str]] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Utilidades de carga
# ═════════════════════════════════════════════════════════════════════════════

def load_vocab_and_weights(wl: int, mode: str) -> tuple[list[str], dict[str, float]]:
    """
    Carga el vocabulario y los pesos desde el CSV oficial.
    Para 'uniform': todos los pesos son 1/N.
    Para 'frequency': importa _sigmoid_weights del framework si está disponible,
    sino aproxima con log-count normalizado.
    """
    csv_path = DATA_DIR / f"spanish_{wl}letter.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}. "
                                f"Corre python3 run_all.py --setup-only primero.")

    words, counts = [], {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["word"].strip().lower()
            c = int(row["count"])
            if len(w) == wl:
                words.append(w)
                counts[w] = c

    if mode == "uniform":
        n = len(words)
        probs = {w: 1.0 / n for w in words}
    else:
        # Intentar usar la función oficial del framework
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from lexicon import _sigmoid_weights  # type: ignore
            raw = _sigmoid_weights(counts, steepness=1.5)
            total = sum(raw.values())
            probs = {w: raw[w] / total for w in words}
        except ImportError:
            # Fallback: log-count normalizado
            import math as _math
            raw = {w: _math.log1p(counts[w]) for w in words}
            total = sum(raw.values())
            probs = {w: raw[w] / total for w in words}

    return words, probs


def load_t2_table(wl: int, mode: str) -> dict[str, str]:
    """
    Carga la tabla T2 precomputada desde JSON.
    Busca en el directorio actual y en la raíz del repo.
    """
    for search_dir in [Path("."), REPO_ROOT]:
        path = search_dir / f"t2_table_{wl}_{mode}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No se encontró t2_table_{wl}_{mode}.json. "
        f"Colócalo en el directorio actual o en {REPO_ROOT}."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Motor de feedback (replica exacta de wordle_env.feedback)
# ═════════════════════════════════════════════════════════════════════════════

def _feedback(secret: str, guess: str) -> tuple[int, ...]:
    """
    Calcula el feedback de Wordle.
    2 = verde, 1 = amarillo, 0 = gris.
    Algoritmo de dos pasadas idéntico al de wordle_env.py.
    """
    n = len(secret)
    result = [0] * n
    secret_counts: dict[str, int] = defaultdict(int)

    # Primera pasada: verdes
    for i in range(n):
        if guess[i] == secret[i]:
            result[i] = 2
        else:
            secret_counts[secret[i]] += 1

    # Segunda pasada: amarillos
    for i in range(n):
        if result[i] == 0 and secret_counts.get(guess[i], 0) > 0:
            result[i] = 1
            secret_counts[guess[i]] -= 1

    return tuple(result)


def _filter_candidates(candidates: list[str], guess: str,
                       pattern: tuple[int, ...]) -> list[str]:
    """Filtra candidatos compatibles con (guess, pattern)."""
    return [w for w in candidates if _feedback(w, guess) == pattern]


# ═════════════════════════════════════════════════════════════════════════════
# Análisis de restricciones posicionales
# ═════════════════════════════════════════════════════════════════════════════

def parse_constraints(opener: str, pat1: tuple[int, ...],
                      guess2: str, pat2: tuple[int, ...]) -> dict:
    """
    Extrae las restricciones del estado del juego tras T1 y T2.

    Retorna un dict con:
      greys:      set de letras confirmadas ausentes
      greens:     dict pos -> letra confirmada
      yellows:    dict letra -> set de posiciones prohibidas (donde salió amarilla)
      known_letters: set de letras que sí están en la palabra (amarillas + verdes)
    """
    greys: set[str] = set()
    greens: dict[int, str] = {}
    yellows: dict[str, set[int]] = defaultdict(set)
    known: set[str] = set()

    for g, pat in [(opener, pat1), (guess2, pat2)]:
        for i, (ch, val) in enumerate(zip(g, pat)):
            if val == 2:
                greens[i] = ch
                known.add(ch)
            elif val == 1:
                yellows[ch].add(i)
                known.add(ch)
            else:  # val == 0
                # Ojo: solo es gris verdadero si la letra no apareció como
                # verde/amarilla EN ESTE MISMO GUESS (letras repetidas).
                # Si ya está en known por otro guess, no la añadimos a greys.
                # Si en este mismo guess la letra salió verde/amarilla en otra
                # posición, el 0 es por exceso de ocurrencias, no ausencia.
                other_vals = [pat[j] for j, c in enumerate(g) if c == ch]
                if not any(v > 0 for v in other_vals):
                    greys.add(ch)

    return {
        "greys":         greys,
        "greens":        greens,
        "yellows":       yellows,
        "known":         known,
    }


def _is_valid_t3_guess(guess: str, constraints: dict) -> bool:
    """
    Verifica que un guess de T3 no desperdicia posiciones conocidas.

    Un guess es INVÁLIDO informativamente si:
      1. Usa una letra gris (ya sabemos que no está, da 0 info)
      2. Pone una letra amarilla en la misma posición donde ya salió amarilla
         (el juego nos daría amarilla de nuevo — info que ya tenemos)
      3. Pone en una posición verde una letra DISTINTA a la verde confirmada
         (viola la restricción conocida — no puede ser la palabra secreta,
         así que tampoco es un buen guess exploratorio salvo casos muy raros)

    Nota: NO prohibimos reusar letras verdes en su posición correcta —
    a veces es necesario en endgame. Pero sí filtramos los más obvios.
    """
    greys   = constraints["greys"]
    greens  = constraints["greens"]
    yellows = constraints["yellows"]

    for i, ch in enumerate(guess):
        # Regla 1: no usar letras grises
        if ch in greys:
            return False
        # Regla 2: no poner amarilla en la misma posición donde ya salió amarilla
        if ch in yellows and i in yellows[ch]:
            return False
        # Regla 3: no contradecir una verde confirmada
        if i in greens and greens[i] != ch:
            return False

    return True


def _count_new_letters(guess: str, opener: str, guess2: str) -> int:
    """Cuenta cuántas letras del guess son completamente nuevas (no en T1 ni T2)."""
    used = set(opener) | set(guess2)
    return sum(1 for ch in set(guess) if ch not in used)


# ═════════════════════════════════════════════════════════════════════════════
# Generación de no-palabras branch-specific
# ═════════════════════════════════════════════════════════════════════════════

def generate_branch_nonwords(candidates: list[str], wl: int,
                             constraints: dict, n: int = 300) -> list[str]:
    """
    Genera no-palabras optimizadas para el branch actual.

    Estrategia:
    1. Identifica letras completamente nuevas (no en grises, no en known)
       — estas son las más informativas porque cada posición da info nueva.
    2. Completa con letras ambiguas del subvocabulario (que distinguen candidatos).
    3. Genera permutaciones respetando las restricciones posicionales.
    4. Descarta las que violan _is_valid_t3_guess.

    El resultado son strings que maximize posibilidades de información
    sin desperdiciar posiciones en letras ya conocidas.
    """
    greys  = constraints["greys"]
    greens = constraints["greens"]
    known  = constraints["known"]

    # Letras completamente vírgenes (no grises, no conocidas)
    all_letters = set("abcdefghijklmnopqrstuvwxyzñ")
    virgin = all_letters - greys - known

    # Frecuencia de letras vírgenes en el subvocabulario
    freq: dict[str, int] = defaultdict(int)
    for w in candidates:
        for ch in set(w):
            if ch in virgin:
                freq[ch] += 1

    # También contar letras ambiguas (amarillas) que varían de posición
    for w in candidates:
        for ch in set(w):
            if ch not in greys:
                freq[ch] += 1  # bonus por ser informativa en este subvocab

    top_letters = sorted(freq, key=lambda c: -freq[c])

    # Pool de letras candidatas para construir no-palabras
    # Priorizar vírgenes, completar con otras si hace falta
    pool = [c for c in top_letters if c in virgin]
    if len(pool) < wl:
        extras = [c for c in top_letters if c not in pool]
        pool.extend(extras[:wl - len(pool) + 4])
    pool = pool[:wl + 6]  # tomar un poco más para tener variedad

    candidate_set = set(candidates)
    non_words: list[str] = []
    seen: set[str] = set()

    # Generar permutaciones de subconjuntos de tamaño wl
    from itertools import combinations
    for size in range(wl, max(wl - 1, wl - 1), -1):
        for combo in combinations(pool, wl):
            for perm in permutations(combo):
                nw = "".join(perm)
                if nw in seen:
                    continue
                seen.add(nw)
                # Debe tener letras únicas y respetar restricciones
                if len(set(nw)) == wl and _is_valid_t3_guess(nw, constraints):
                    non_words.append(nw)
                if len(non_words) >= n:
                    return non_words

    return non_words


# ═════════════════════════════════════════════════════════════════════════════
# Cálculo de entropía
# ═════════════════════════════════════════════════════════════════════════════

def _entropy(guess: str, candidates: list[str],
             weights: dict[str, float]) -> float:
    """
    Entropía de Shannon ponderada de la partición que produce `guess`
    sobre `candidates` con distribución `weights`.
    H = -Σ p(f) * log2(p(f))
    """
    partition: dict[tuple, float] = defaultdict(float)
    for w in candidates:
        fb = _feedback(w, guess)
        partition[fb] += weights.get(w, 0.0)

    total = sum(partition.values())
    if total == 0:
        return 0.0

    h = 0.0
    for mass in partition.values():
        if mass > 0:
            p = mass / total
            h -= p * math.log2(p)
    return h


def _normalize(candidates: list[str], probs: dict[str, float]) -> dict[str, float]:
    """Renormaliza probabilidades al subconjunto de candidatos."""
    total = sum(probs.get(w, 0.0) for w in candidates)
    if total == 0:
        n = len(candidates)
        return {w: 1.0 / n for w in candidates}
    return {w: probs.get(w, 0.0) / total for w in candidates}


# ═════════════════════════════════════════════════════════════════════════════
# Búsqueda del mejor T3 para un branch específico
# ═════════════════════════════════════════════════════════════════════════════

def best_t3_for_branch(candidates: list[str],
                       vocab: list[str],
                       probs: dict[str, float],
                       opener: str, pat1: tuple[int, ...],
                       guess2: str, pat2: tuple[int, ...],
                       wl: int) -> str:
    """
    Encuentra el mejor guess T3 para un branch dado.

    Pool de evaluación:
      - Vocabulario completo (palabras reales del juego)
      - No-palabras branch-specific (respetan restricciones posicionales)

    Filtro: solo evalúa guesses que no desperdician posiciones conocidas.

    Criterio: máxima entropía. Desempate: más letras nuevas → menor léxico.
    """
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        # Con 2 candidatos, adivinar el más probable directamente
        return max(candidates, key=lambda w: probs.get(w, 0.0))

    constraints = parse_constraints(opener, pat1, guess2, pat2)
    weights = _normalize(candidates, probs)

    # Pool: vocabulario completo válido + no-palabras branch-specific
    valid_vocab = [w for w in vocab if _is_valid_t3_guess(w, constraints)]
    nonwords    = generate_branch_nonwords(candidates, wl, constraints, n=300)
    pool = valid_vocab + [nw for nw in nonwords if nw not in set(valid_vocab)]

    # Si el filtro es muy estricto y no queda nada, usar candidatos directamente
    if not pool:
        pool = candidates

    best_guess  = candidates[0]
    best_H      = -1.0
    best_new    = -1

    for g in pool:
        h = _entropy(g, candidates, weights)
        new = _count_new_letters(g, opener, guess2)

        # Selección: mayor H, desempate por más letras nuevas, desempate final léxico
        if (h > best_H + 1e-9
                or (abs(h - best_H) < 1e-9 and new > best_new)
                or (abs(h - best_H) < 1e-9 and new == best_new and g < best_guess)):
            best_H     = h
            best_new   = new
            best_guess = g

    return best_guess


# ═════════════════════════════════════════════════════════════════════════════
# Worker para multiprocessing (procesa todos los sub-branches de un pat_T1)
# ═════════════════════════════════════════════════════════════════════════════

def _worker(args: tuple) -> tuple[str, dict[str, str]]:
    """
    Procesa todos los sub-branches de un patrón T1 dado.

    Retorna (pat1_str, {key: best_guess}) donde key = "pat1|guess2|pat2".
    """
    (pat1_str, pat1, guess2,
     vocab, probs, opener, wl, min_cands) = args

    results: dict[str, str] = {}

    # Candidatos tras T1
    cands_after_t1 = _filter_candidates(vocab, opener, pat1)
    if not cands_after_t1:
        return pat1_str, results

    # Todos los patrones posibles de T2
    pat2_counts: dict[tuple, list[str]] = defaultdict(list)
    for w in cands_after_t1:
        fb = _feedback(w, guess2)
        pat2_counts[fb].append(w)

    for pat2, cands_after_t2 in pat2_counts.items():
        if len(cands_after_t2) <= min_cands:
            # Trivial: ≤2 candidatos no necesitan lookup
            continue

        pat2_str = "".join(str(x) for x in pat2)
        key = f"{pat1_str}|{guess2}|{pat2_str}"

        best = best_t3_for_branch(
            candidates=cands_after_t2,
            vocab=vocab,
            probs=probs,
            opener=opener,
            pat1=pat1,
            guess2=guess2,
            pat2=pat2,
            wl=wl,
        )
        results[key] = best

    return pat1_str, results


# ═════════════════════════════════════════════════════════════════════════════
# Función principal de precomputación
# ═════════════════════════════════════════════════════════════════════════════

def precompute_t3(wl: int, mode: str, workers: int, min_cands: int = 2) -> dict[str, str]:
    """
    Precomputa la tabla T3 completa para (wl, mode).

    min_cands: estados con ≤ min_cands candidatos se omiten (triviales).
    """
    print(f"\n{'═' * 60}")
    print(f"  T3 precomputación: {wl}-letras | modo: {mode}")
    print(f"{'═' * 60}")

    vocab, probs = load_vocab_and_weights(wl, mode)
    t2_table     = load_t2_table(wl, mode)
    opener       = OPENERS[wl][mode]

    print(f"  Vocabulario:  {len(vocab)} palabras")
    print(f"  Opener:       '{opener}'")
    print(f"  Entradas T2:  {len(t2_table)}")
    print(f"  Workers:      {workers}")

    # Construir tareas: una por patrón T1 con su guess2 correspondiente
    tasks = []
    all_pat1: dict[str, list[str]] = defaultdict(list)

    for w in vocab:
        fb = _feedback(w, opener)
        all_pat1["".join(str(x) for x in fb)].append(w)

    for pat1_str, guess2 in t2_table.items():
        if pat1_str not in all_pat1:
            continue  # branch vacío (no alcanzable)
        pat1 = tuple(int(x) for x in pat1_str)
        tasks.append((
            pat1_str, pat1, guess2,
            vocab, probs, opener, wl, min_cands
        ))

    print(f"  Branches T1 a procesar: {len(tasks)}")
    t_start = time.monotonic()

    # Ejecutar en paralelo
    table: dict[str, str] = {}
    completed = 0

    with Pool(processes=workers) as pool:
        for pat1_str, branch_results in pool.imap_unordered(_worker, tasks):
            table.update(branch_results)
            completed += 1
            elapsed = time.monotonic() - t_start
            eta = (elapsed / completed) * (len(tasks) - completed) if completed else 0
            print(f"  [{completed:3d}/{len(tasks)}] pat1={pat1_str} "
                  f"→ {len(branch_results):3d} entradas T3  "
                  f"| {elapsed:.0f}s transcurridos, ETA ~{eta:.0f}s",
                  flush=True)

    total_time = time.monotonic() - t_start
    print(f"\n  ✓ Completado en {total_time:.1f}s")
    print(f"  Total entradas T3: {len(table)}")

    return table


# ═════════════════════════════════════════════════════════════════════════════
# Guardar y reporte
# ═════════════════════════════════════════════════════════════════════════════

def save_table(table: dict[str, str], wl: int, mode: str) -> None:
    path = Path(f"t3_table_{wl}_{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"  → Guardado en: {path}  ({path.stat().st_size // 1024} KB)")


def print_sample(table: dict[str, str], n: int = 10) -> None:
    """Muestra algunas entradas para validación visual."""
    print(f"\n  Muestra de {min(n, len(table))} entradas:")
    for i, (k, v) in enumerate(sorted(table.items())[:n]):
        parts = k.split("|")
        pat1, g2, pat2 = parts[0], parts[1], parts[2]
        is_word = not any(c.isdigit() for c in v)
        kind = "PALABRA" if is_word else "NO-PAL "
        print(f"    pat1={pat1} → T2='{g2}' → pat2={pat2}  ⇒  T3='{v}' [{kind}]")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Precomputa tabla T3 para Wordle español")
    parser.add_argument("--length", choices=["4", "5", "6", "all"], default="4",
                        help="Longitud de palabra a precomputar (default: 4)")
    parser.add_argument("--mode", choices=["uniform", "frequency", "both"], default="both",
                        help="Modo de probabilidad (default: both)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help="Número de workers paralelos (default: núcleos - 1)")
    parser.add_argument("--min-cands", type=int, default=2,
                        help="Mínimo de candidatos para incluir en tabla (default: 2)")
    args = parser.parse_args()

    lengths = [4, 5, 6] if args.length == "all" else [int(args.length)]
    modes   = ["uniform", "frequency"] if args.mode == "both" else [args.mode]

    for wl in lengths:
        for mode in modes:
            # Verificar que existe la tabla T2
            found = False
            for search_dir in [Path("."), REPO_ROOT]:
                if (search_dir / f"t2_table_{wl}_{mode}.json").exists():
                    found = True
                    break
            if not found:
                print(f"\n⚠  Saltando {wl}-letras {mode}: "
                      f"no se encontró t2_table_{wl}_{mode}.json")
                continue

            table = precompute_t3(wl, mode, workers=args.workers,
                                   min_cands=args.min_cands)
            save_table(table, wl, mode)
            print_sample(table, n=8)

    print("\n✓ Precomputación T3 finalizada.")


if __name__ == "__main__":
    main()
