"""
OPTIMAL DP SOLVER — paralelo por ramas del opener
══════════════════════════════════════════════════

Paraleliza el DP dividiendo las ramas post-opener entre workers.

Después del opener, el vocabulario se divide en N ramas independientes:
  4 letras: ~56 ramas  → speedup ~min(56, n_workers)
  5 letras: ~170 ramas → speedup ~min(170, n_workers)
  6 letras: ~438 ramas → speedup ~min(438, n_workers)

Cada rama es completamente independiente — no comparte estado.
Cada worker tiene su propio cache de memoización local.

Tiempos estimados con 8 workers:
  4 letras: 4-8h → ~30-60 min
  5 letras: ~24h  → ~3-4 horas
  6 letras: días  → ~12-24 horas (no recomendado aún)

Uso:
  python3 parallel_dp.py --length 4
  python3 parallel_dp.py --length 5
  python3 parallel_dp.py --length 4 --workers 8
  python3 parallel_dp.py --length 4 --mode uniform
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

try:
    from wordle_env import feedback, filter_candidates
    from lexicon import _sigmoid_weights
    print("✓ framework importado")
except ImportError:
    raise SystemExit("ERROR: corre desde torneo_wordle/")

WIN_PAT_4 = (2, 2, 2, 2)
WIN_PAT_5 = (2, 2, 2, 2, 2)
WIN_PAT_6 = (2, 2, 2, 2, 2, 2)
OPENERS = {
    4: {'uniform': 'aore',   'frequency': 'aore'},
    5: {'uniform': 'sareo',  'frequency': 'sareo'},  # ← se llena cuando termina el search
    6: {'uniform': 'xxxxxx', 'frequency': 'yyyyyy'},
}
MAX_DEPTH = 6


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


def generate_non_words(vocab, wl, n=100):
    """
    Carga no-palabras desde nonwords_pool_{wl}letter.json si existe.
    Si no existe, usa generación heurística estándar.
    """
    json_path = Path(f"nonwords_pool_{wl}letter.json")
    if json_path.exists():
        with open(json_path, encoding='utf-8') as f:
            non_words = json.load(f)
        vocab_set = set(vocab)
        non_words = [w for w in non_words if w not in vocab_set]
        print(f"  ✓ No-palabras cargadas desde {json_path}: {len(non_words)}")
        return non_words

    print(f"  (usando no-palabras heurísticas — corre generate_nonwords.py para mejorar)")
    vocab_set = set(vocab)
    SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")
    pos_freq = [defaultdict(float) for _ in range(wl)]
    w_u = 1.0 / len(vocab)
    for w in vocab:
        for i, ch in enumerate(w):
            pos_freq[i][ch] += w_u
    overall = defaultdict(float)
    for i in range(wl):
        for ch, f in pos_freq[i].items():
            overall[ch] += f
    top = [ch for ch, _ in sorted(overall.items(), key=lambda x: -x[1])]
    pool = top[:min(wl + 8, len(top))]
    non_words, seen = [], set()
    for combo in itertools.combinations(pool, wl):
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


def generate_branch_nonwords(opener, pat_tuple, vocab, wl, n=150):
    """
    Genera no-palabras específicas para una rama del opener.

    El truco: para cada rama sabemos exactamente qué letras son grises
    (ausentes), amarillas (presentes, posición incorrecta) y verdes
    (presentes, posición correcta). Las no-palabras óptimas para esta
    rama deben:

    1. Evitar letras grises (ya sabemos que no están — usarlas desperdicia
       información)
    2. Incluir letras amarillas en posiciones distintas (confirmar posición)
    3. Usar letras completamente nuevas en las posiciones restantes

    Esto genera no-palabras mucho más informativas que las globales para
    las ramas donde muchas letras del opener son grises.
    """
    SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")
    vocab_set = set(vocab)

    # Analizar qué sabemos del opener
    grey_letters   = set()  # definitivamente ausentes
    yellow_letters = set()  # presentes pero posición incorrecta
    green_letters  = set()  # presentes y posición correcta
    green_positions = {}    # pos → letra

    for i, (ch, fb) in enumerate(zip(opener, pat_tuple)):
        if fb == 0:
            grey_letters.add(ch)
        elif fb == 1:
            yellow_letters.add(ch)
        else:
            green_letters.add(ch)
            green_positions[i] = ch

    known_letters = green_letters | yellow_letters

    # Letras disponibles: no grises, ordenadas por frecuencia en el vocab
    letter_freq: dict[str, float] = defaultdict(float)
    for w in vocab:
        for ch in set(w):
            letter_freq[ch] += 1.0 / len(vocab)

    # Letras nuevas (no en opener): máxima información
    new_letters = sorted(
        [ch for ch in SPANISH_LETTERS
         if ch not in grey_letters and ch not in known_letters],
        key=lambda c: -letter_freq[c]
    )

    # Letras reusables (amarillas/verdes, excluidas grises)
    reusable = sorted(
        [ch for ch in known_letters],
        key=lambda c: -letter_freq[c]
    )

    non_words, seen = [], set()

    # Estrategia 1: palabras con solo letras NUEVAS (máxima información nueva)
    # — Ideales cuando muchas letras del opener son grises
    if len(new_letters) >= wl:
        for combo in itertools.combinations(new_letters[:12], wl):
            for perm in itertools.permutations(combo):
                # Respetar posiciones verdes si las hay
                valid = True
                for pos, ch in green_positions.items():
                    if perm[pos] != ch:
                        # No necesitamos respetar verdes en no-palabras,
                        # pero es más informativo testear otras posiciones
                        pass
                nw = ''.join(perm)
                if (len(set(nw)) == wl
                        and nw not in seen
                        and nw not in vocab_set):
                    seen.add(nw)
                    non_words.append(nw)
            if len(non_words) >= n // 2:
                break

    # Estrategia 2: mezcla de letras nuevas + amarillas en nuevas posiciones
    # — Ideales para confirmar posición de letras amarillas
    if yellow_letters and len(new_letters) >= wl - len(yellow_letters):
        yellow_list = sorted(yellow_letters, key=lambda c: -letter_freq[c])
        new_fill = new_letters[:wl + 4]

        for n_yellow in range(1, min(len(yellow_list) + 1, wl)):
            for y_combo in itertools.combinations(yellow_list[:4], n_yellow):
                needed = wl - n_yellow
                for fill in itertools.combinations(new_fill, needed):
                    if any(ch in y_combo for ch in fill):
                        continue
                    letters = list(y_combo) + list(fill)
                    for perm in itertools.permutations(letters):
                        # Verificar que amarillas no están en sus posiciones
                        # originales (sería feedback inútil)
                        valid = True
                        for pos, ch in enumerate(opener):
                            if pat_tuple[pos] == 1 and perm[pos] == ch:
                                valid = False
                                break
                        if not valid:
                            continue
                        nw = ''.join(perm)
                        if (len(set(nw)) == wl
                                and nw not in seen
                                and nw not in vocab_set):
                            seen.add(nw)
                            non_words.append(nw)
                        if len(non_words) >= n:
                            return non_words
                    if len(non_words) >= n:
                        return non_words

    return non_words[:n]


# ══════════════════════════════════════════════════════════════════════════════
# DP recursivo con memoización local (por worker)
# ══════════════════════════════════════════════════════════════════════════════

class BranchSolver:
    """
    Resuelve el subárbol DP para una sola rama del opener.
    Tiene su propio cache de memoización — no necesita coordinación
    con otros workers.
    """

    def __init__(self, vocab, weights, full_pool, wl):
        self.vocab     = vocab
        self.weights   = weights
        self.full_pool = full_pool  # vocab + non_words
        self.wl        = wl
        self.win_pat   = tuple([2] * wl)
        self._memo: dict[frozenset, tuple[float, str]] = {}
        self.n_states = 0

    def _normalize(self, candidates):
        raw = {w: self.weights.get(w, 1e-10) for w in candidates}
        total = sum(raw.values())
        if total == 0:
            return {w: 1.0 / len(candidates) for w in candidates}
        return {w: v / total for w, v in raw.items()}

    def _get_pool(self, depth, candidates):
        """
        Siempre usamos el pool completo (vocab + non_words) sin excepción.

        No hay aproximación — el guess óptimo puede estar fuera del conjunto
        de candidatos en cualquier profundidad (el caso PATO/GATO/MATO donde
        una no-palabra es mejor que cualquier candidato es un ejemplo real).

        El costo extra en depth 3+ es mínimo: con 3-8 candidatos restantes,
        evaluar ~1,953 palabras sobre ellos toma microsegundos por estado.
        La memoización garantiza que cada estado único se resuelve solo una vez.
        """
        return self.full_pool

    def solve(self, candidates, depth=0):
        """
        V*(S) = min_g Σ_f p(f|g,S) × (1 + V*(S_f))

        Retorna (score_esperado, mejor_guess).
        """
        n = len(candidates)

        if n == 0:
            return 0.0, ''
        if n == 1:
            return 1.0, candidates[0]
        if n == 2:
            best = max(candidates, key=lambda w: self.weights.get(w, 0.0))
            return 1.5, best
        if depth >= MAX_DEPTH:
            # Sin intentos restantes — penalización alta
            return float(MAX_DEPTH + 1), candidates[0]

        state_key = frozenset(candidates)
        if state_key in self._memo:
            return self._memo[state_key]

        self.n_states += 1
        pool = self._get_pool(depth, candidates)
        w = self._normalize(candidates)

        best_guess = candidates[0]
        best_score = float('inf')

        for g in pool:
            partition: dict[tuple, list] = defaultdict(list)
            for cand in candidates:
                partition[feedback(cand, g)].append(cand)

            score = 0.0
            for pat, group in partition.items():
                p_f = sum(w.get(c, 0.0) for c in group)
                if pat == self.win_pat:
                    score += p_f * 1.0
                elif len(group) == 1:
                    score += p_f * 2.0
                elif len(group) == 2:
                    score += p_f * 2.5
                else:
                    sub_score, _ = self.solve(group, depth + 1)
                    score += p_f * (1.0 + sub_score)

                # Poda: si ya superamos el mejor, abandonar este guess
                if score >= best_score:
                    break

            if score < best_score:
                best_score = score
                best_guess = g

        result = (best_score, best_guess)
        self._memo[state_key] = result
        return result

    def build_subtree(self, candidates, guess, depth=0):
        """Construye el árbol explícito para este subconjunto."""
        node = {"g": guess, "d": depth}
        if len(candidates) <= 1:
            return node

        partition: dict[tuple, list] = defaultdict(list)
        for cand in candidates:
            partition[feedback(cand, guess)].append(cand)

        children = {}
        for pat, group in sorted(partition.items()):
            pat_str = ''.join(str(x) for x in pat)
            if pat == self.win_pat or not group:
                continue
            if len(group) == 1:
                children[pat_str] = {"g": group[0], "d": depth + 1}
                continue
            _, next_guess = self.solve(group, depth + 1)
            if next_guess is None:
                next_guess = group[0]
            subtree = self.build_subtree(group, next_guess, depth + 1)
            children[pat_str] = subtree

        if children:
            node["c"] = children
        return node


# ══════════════════════════════════════════════════════════════════════════════
# Worker function (top-level para que pickle funcione con multiprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def _solve_branch(args):
    """
    Resuelve una rama completa del opener.
    args = (pat1_str, pat1_tuple, branch_candidates, vocab, weights,
            global_non_words, wl, opener)

    Genera no-palabras branch-specific que evitan letras grises del opener.
    Para pat=0100 (a,r,e grises), el pool no contendrá palabras con a,r,e
    — eliminando el problema de elegir 'mana' o 'aonm'.
    """
    pat1_str, pat1_tuple, branch_candidates, vocab, weights, \
        global_non_words, wl, opener = args

    t0 = time.monotonic()

    # Pool branch-specific: vocab + no-palabras que respetan lo que sabemos
    branch_nw  = generate_branch_nonwords(opener, pat1_tuple, vocab, wl, n=150)
    # Combinar global + branch, deduplicar
    seen_pool  = set(vocab)
    extra_nw   = []
    for nw in branch_nw + global_non_words:
        if nw not in seen_pool:
            seen_pool.add(nw)
            extra_nw.append(nw)
    branch_pool = vocab + extra_nw

    solver = BranchSolver(vocab, weights, branch_pool, wl)

    # Resolver el subárbol de esta rama
    score, best_t2 = solver.solve(branch_candidates, depth=1)

    # Construir árbol explícito
    subtree = solver.build_subtree(branch_candidates, best_t2, depth=1)

    elapsed = time.monotonic() - t0
    return {
        "pat1":       pat1_str,
        "n_cands":    len(branch_candidates),
        "score":      score,
        "best_t2":    best_t2,
        "subtree":    subtree,
        "n_states":   solver.n_states,
        "elapsed":    elapsed,
        "pool_size":  len(branch_pool),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

def run(wl, mode, n_workers):
    print(f"\n{'═'*60}")
    print(f"  DP ÓPTIMO — {wl} letras | modo: {mode} | workers: {n_workers}")
    print(f"{'═'*60}")

    vocab, weights_u, weights_f = load_vocab(wl)
    weights = weights_u if mode == "uniform" else weights_f
    non_words = generate_non_words(vocab, wl, n=100)
    opener = OPENERS[wl][mode]
    win_pat = tuple([2] * wl)

    print(f"  Opener: '{opener}'")
    print(f"  Vocab: {len(vocab)}, no-palabras globales: {len(non_words)}")
    print(f"  Cada rama recibirá su pool branch-specific adicional (~150 nw)")

    # Dividir vocabulario en ramas post-opener
    branches: dict[tuple, list] = defaultdict(list)
    for w in vocab:
        branches[feedback(w, opener)].append(w)

    # Ordenar ramas por tamaño descendente (las más grandes primero)
    # para mejor balance de carga entre workers
    branch_list = sorted(branches.items(), key=lambda x: -len(x[1]))
    n_branches = len(branch_list)

    print(f"  Ramas post-opener: {n_branches}")
    print(f"  Rama más grande: {branch_list[0][1][:3]}... "
          f"({len(branch_list[0][1])} candidatos)")
    print(f"  Ramas triviales (≤2 cands): "
          f"{sum(1 for _, c in branch_list if len(c) <= 2)}")

    # Preparar argumentos para workers
    tasks = []
    for pat, cands in branch_list:
        pat_str = ''.join(str(x) for x in pat)
        if pat == win_pat:
            continue
        tasks.append((pat_str, pat, cands, vocab, weights,
                      non_words, wl, opener))

    print(f"\n  Lanzando {len(tasks)} tareas en {n_workers} workers...")
    print(f"  (progreso se reporta conforme terminan ramas)")
    sys.stdout.flush()

    # Correr en paralelo
    t0 = time.monotonic()
    results_by_pat = {}
    completed = 0
    total_states = 0

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_solve_branch, tasks, chunksize=1):
            completed += 1
            total_states += result["n_states"]
            results_by_pat[result["pat1"]] = result

            elapsed = time.monotonic() - t0
            eta = elapsed / completed * (len(tasks) - completed)
            print(f"  [{completed:3d}/{len(tasks)}] "
                  f"pat={result['pat1']}  "
                  f"cands={result['n_cands']:3d}  "
                  f"pool={result['pool_size']}  "
                  f"t2='{result['best_t2']}'  "
                  f"score={result['score']:.3f}  "
                  f"states={result['n_states']:,}  "
                  f"ETA={eta/60:.1f}min")
            sys.stdout.flush()

    total_elapsed = time.monotonic() - t0
    print(f"\n  ✓ Completado en {total_elapsed/60:.1f} min")
    print(f"  Estados únicos totales: {total_states:,}")

    # ── Construir árbol completo ──────────────────────────────────────────────
    # El nodo raíz es el opener
    root = {
        "g": opener,
        "d": 0,
        "c": {}
    }

    # Añadir rama "ganada en T1" si aplica
    win_pat_str = ''.join(str(x) for x in win_pat)
    # (si el opener == secreto, ya ganamos, no hay rama)

    for pat, result in results_by_pat.items():
        root["c"][pat] = result["subtree"]

    # ── Estadísticas ──────────────────────────────────────────────────────────
    stats = _tree_stats(root)
    print(f"\n  DISTRIBUCIÓN DE GUESSES:")
    total_words = sum(stats.values())
    weighted_mean = sum(d * n for d, n in stats.items()) / total_words
    for d in sorted(stats):
        pct = stats[d] / total_words * 100
        print(f"    {d} guesses: {stats[d]:4d} ({pct:.1f}%)")
    print(f"  Mean: {weighted_mean:.4f} guesses")
    print(f"  Solve rate ≤6: "
          f"{sum(n for d,n in stats.items() if d<=6)/total_words*100:.2f}%")

    # ── Guardar ───────────────────────────────────────────────────────────────
    # Árbol completo (para debug/inspección)
    tree_path = Path(f"optimal_tree_{wl}_{mode}.json")
    with open(tree_path, 'w', encoding='utf-8') as f:
        json.dump(root, f, ensure_ascii=False, separators=(',', ':'))
    print(f"\n  Árbol guardado: {tree_path} "
          f"({tree_path.stat().st_size/1024:.0f} KB)")

    # Versión plana para lookup O(1) en strategy.py
    flat = _flatten_tree(root)
    flat_path = Path(f"optimal_flat_{wl}_{mode}.json")
    with open(flat_path, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, separators=(',', ':'))
    print(f"  Lookup plano:  {flat_path} "
          f"({flat_path.stat().st_size/1024:.0f} KB)")
    print(f"  Entradas en lookup: {len(flat):,}")

    return weighted_mean, stats


def _tree_stats(tree):
    """Cuenta palabras por profundidad de resolución."""
    stats: dict[int, int] = defaultdict(int)
    def _walk(node, depth):
        if "c" not in node or not node["c"]:
            stats[depth] += 1
            return
        for child in node["c"].values():
            _walk(child, depth + 1)
    _walk(tree, 0)
    return dict(stats)


def _flatten_tree(tree):
    """
    Convierte árbol anidado a dict plano para lookup O(1).

    Clave: "pat1|pat2|pat3|..."
      donde cada pat_i es el feedback string del intento i-1
    Valor: mejor guess para ese estado

    Ejemplo:
      ""           → "aore"    (opener, estado inicial)
      "0120"       → "luis"    (después de opener con feedback 0120)
      "0120|0201"  → "xxxx"    (después de opener→0120, luis→0201)
    """
    flat = {}

    def _walk(node, path):
        flat[path] = node["g"]
        if "c" not in node:
            return
        for pat_str, child in node["c"].items():
            sep = "|" if path else ""
            _walk(child, path + sep + pat_str)

    _walk(tree, "")
    return flat


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DP óptimo paralelo para Wordle español")
    parser.add_argument("--length", type=int, default=4,
                        choices=[4, 5, 6],
                        help="Longitud de palabras")
    parser.add_argument("--mode", default="both",
                        choices=["uniform", "frequency", "both"],
                        help="Modo de probabilidades")
    parser.add_argument("--workers", type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Workers paralelos (default: núcleos-1)")
    args = parser.parse_args()

    print("═" * 60)
    print("  OPTIMAL DP SOLVER — paralelo por ramas")
    print("═" * 60)
    print(f"  CPU cores: {mp.cpu_count()}  |  Workers: {args.workers}")
    print(f"  Longitud: {args.length}  |  Modo: {args.mode}")

    modes = ["uniform", "frequency"] if args.mode == "both" else [args.mode]

    for mode in modes:
        mean, stats = run(args.length, mode, args.workers)
        print(f"\n  ── Resultado {args.length}-letras {mode}: "
              f"mean={mean:.4f} ──")

    print(f"\n{'═'*60}")
    print("  SIGUIENTE PASO:")
    print(f"  cp optimal_flat_{args.length}_uniform.json   "
          f"estudiantes/gabriel_regina/")
    print(f"  cp optimal_flat_{args.length}_frequency.json "
          f"estudiantes/gabriel_regina/")
    print("  El strategy.py los carga automáticamente.")


if __name__ == "__main__":
    mp.freeze_support()
    main()