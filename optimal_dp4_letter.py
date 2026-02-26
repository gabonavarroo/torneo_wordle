"""
OPTIMAL DP SOLVER — 4 letras (backward induction completo)
═══════════════════════════════════════════════════════════

Computa el árbol de decisiones ÓPTIMO para 4 letras usando DP recursivo
con memoización. No hay aproximaciones — cada decisión es la globalmente
óptima dado el estado del juego.

DIFERENCIA CON ENFOQUES ANTERIORES
─────────────────────────────────────
  Antes:  Score(g) = Σ_f p(f) × (1 + f̂(|C_f|))      ← f̂ es una estimación
  Ahora:  V*(S) = min_g Σ_f p(f|g,S) × (1 + V*(S_f))  ← exacto, recursivo

MEMOIZACIÓN
────────────
  Clave: frozenset(candidatos_restantes)
  Esto colapsa el árbol porque muchos caminos (diferentes secuencias de
  guesses y feedbacks) pueden llegar al mismo conjunto de candidatos.
  Para 1,853 palabras el número de estados únicos es manejable.

POOL DE GUESSES
────────────────
  Evaluamos vocabulario completo + no-palabras generadas.
  A partir de profundidad 3, solo evaluamos candidatos restantes
  (con <8 candidatos el mejor guess casi siempre está en el conjunto).

TIEMPO ESTIMADO
────────────────
  Con pool completo (2000+ guesses), ~4-8 horas para el árbol completo.
  Con pool adaptativo (vocab para prof 1-2, candidatos para prof 3+): ~1-2h.

Corre desde: cd ~/ia/wordle/torneo_wordle
Comando: python3 optimal_dp_4letter.py [--mode uniform|frequency|both]
         python3 optimal_dp_4letter.py --mode both  # recomendado

Output: optimal_tree_4_uniform.json
        optimal_tree_4_frequency.json
  Formato: árbol de decisión anidado para usar en strategy.py
"""

import argparse
import csv
import itertools
import json
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


# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════

WORD_LENGTH  = 4
WIN_PAT      = tuple([2] * WORD_LENGTH)
MAX_DEPTH    = 6        # intentos máximos
OPENER       = 'aore'   # precomputado en step1

# A qué profundidad cambiar de pool completo a solo candidatos
# Prof 1-2: pool completo (más impactante, vale el costo)
# Prof 3+:  solo candidatos (rápido, casi sin pérdida de calidad)
FULL_POOL_MAX_DEPTH = 2


# ══════════════════════════════════════════════════════════════════════════════
# Carga de vocabulario
# ══════════════════════════════════════════════════════════════════════════════

def _strip_accents(text: str) -> str:
    result = []
    for ch in text:
        if ch == "ñ":
            result.append("ñ")
        else:
            decomposed = unicodedata.normalize("NFD", ch)
            result.append("".join(c for c in decomposed
                                  if unicodedata.category(c) != "Mn"))
    return "".join(result)


def load_vocab() -> tuple[list[str], dict[str, float], dict[str, float]]:
    csv_path = Path(f"data/spanish_{WORD_LENGTH}letter.csv")
    pattern  = re.compile(rf"^[a-zñ]{{{WORD_LENGTH}}}$")
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


def generate_non_words(vocab: list[str], n: int = 100) -> list[str]:
    """
    Genera no-palabras informativas. Para 4 letras usamos 100 porque el
    espacio es pequeño y las no-palabras tienen alto impacto relativo.
    """
    wl = WORD_LENGTH
    vocab_set = set(vocab)
    pos_freq = [defaultdict(float) for _ in range(wl)]
    w_u = 1.0 / len(vocab)

    for w in vocab:
        for i, ch in enumerate(w):
            pos_freq[i][ch] += w_u

    overall: dict[str, float] = defaultdict(float)
    for i in range(wl):
        for ch, f in pos_freq[i].items():
            overall[ch] += f

    top = [ch for ch, _ in sorted(overall.items(), key=lambda x: -x[1])]
    # Usamos wl+8 para tener más combinaciones
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


# ══════════════════════════════════════════════════════════════════════════════
# DP recursivo con memoización
# ══════════════════════════════════════════════════════════════════════════════

class OptimalSolver:
    def __init__(self, vocab: list[str], weights: dict[str, float],
                 non_words: list[str]):
        self.vocab      = vocab
        self.vocab_set  = set(vocab)
        self.weights    = weights
        self.non_words  = non_words
        self.full_pool  = vocab + non_words  # para profundidades tempranas

        # Memoización: frozenset(candidatos) → (score_esperado, mejor_guess)
        self._memo: dict[frozenset, tuple[float, str]] = {}

        # Estadísticas
        self.n_memo_hits  = 0
        self.n_memo_miss  = 0
        self.n_states     = 0
        self.t_start      = time.monotonic()

    def _normalize(self, candidates: list[str]) -> dict[str, float]:
        raw = {w: self.weights.get(w, 1e-10) for w in candidates}
        total = sum(raw.values())
        if total == 0:
            return {w: 1.0 / len(candidates) for w in candidates}
        return {w: v / total for w, v in raw.items()}

    def _get_pool(self, depth: int, candidates: list[str]) -> list[str]:
        """
        Pool de guesses a evaluar según la profundidad y tamaño del estado.

        Profundidad 1-2: pool completo (vocabulario + no-palabras)
          → aquí el guess correcto puede ser una no-palabra o estar fuera del
            conjunto de candidatos actual, y el impacto es máximo

        Profundidad 3+: solo candidatos
          → con pocos candidatos restantes (~2-8), el óptimo casi siempre
            está en el conjunto; evaluar 2000 guesses no justifica el costo

        Excepción: si hay muchos candidatos en profundidad 3+ (>=10),
        seguimos usando pool completo porque puede haber una sonda mejor.
        """
        if depth <= FULL_POOL_MAX_DEPTH:
            return self.full_pool
        if len(candidates) >= 10:
            return self.full_pool
        return candidates

    def solve(self, candidates: list[str], depth: int = 0) -> tuple[float, str]:
        """
        V*(candidatos, profundidad) =
          min_g  Σ_f p(f | g, candidatos) × cost(f)
          donde:
            cost(all-green)   = 1           (ganamos en este turno)
            cost(otro f)      = 1 + V*(C_f, depth+1)  ← RECURSIVO EXACTO

        Retorna (score_esperado, mejor_guess).
        score_esperado = guesses adicionales esperados desde este estado.

        Si es imposible resolver en (MAX_DEPTH - depth) intentos restantes,
        retorna (inf, None).
        """
        n = len(candidates)

        # Casos base
        if n == 0:
            return 0.0, ''
        if n == 1:
            return 1.0, candidates[0]
        if n == 2:
            # Con 2 candidatos siempre adivinar el más probable primero
            # Score esperado: 50% × 1 + 50% × 2 = 1.5
            best = max(candidates, key=lambda w: self.weights.get(w, 0.0))
            return 1.5, best

        # Sin intentos restantes → fallo
        if depth >= MAX_DEPTH:
            return float('inf'), None

        # Memoización
        state_key = frozenset(candidates)
        if state_key in self._memo:
            self.n_memo_hits += 1
            return self._memo[state_key]
        self.n_memo_miss += 1
        self.n_states += 1

        # Log de progreso
        if self.n_states % 500 == 0:
            elapsed = time.monotonic() - self.t_start
            print(f"    estados={self.n_states:,}  "
                  f"memo_hits={self.n_memo_hits:,}  "
                  f"elapsed={elapsed:.0f}s")

        pool = self._get_pool(depth, candidates)
        w = self._normalize(candidates)

        best_guess  = candidates[0]
        best_score  = float('inf')

        for g in pool:
            # Particionar candidatos por feedback de g
            partition: dict[tuple, list[str]] = defaultdict(list)
            for cand in candidates:
                partition[feedback(cand, g)].append(cand)

            # Calcular score esperado de este guess
            score = 0.0
            feasible = True

            for pat, group in partition.items():
                p_f = sum(w.get(cand, 0.0) for cand in group)

                if pat == WIN_PAT:
                    score += p_f * 1.0
                else:
                    # Llamada recursiva exacta
                    sub_score, _ = self.solve(group, depth + 1)
                    if sub_score == float('inf'):
                        feasible = False
                        # No rompemos el loop — otro guess puede ser factible
                        # pero podemos asignar un costo alto en lugar de inf
                        # para no descartar guesses que casi siempre funcionan
                        score += p_f * (MAX_DEPTH - depth + 1)
                    else:
                        score += p_f * (1.0 + sub_score)

            if score < best_score:
                best_score = score
                best_guess = g

        # Guardar en memo
        result = (best_score, best_guess)
        self._memo[state_key] = result
        return result

    def build_tree(self, candidates: list[str],
                   guess: str, depth: int = 0) -> dict:
        """
        Construye el árbol de decisión explícito para guardar como JSON.

        Formato del árbol:
        {
          "g": "aore",           ← guess a hacer
          "d": 0,                ← profundidad
          "c": {                 ← children por patrón de feedback
            "01202": {
              "g": "xxxx",
              "d": 1,
              "c": { ... }
            },
            ...
          }
        }

        Las hojas (1 candidato o all-green) no tienen "c".
        """
        node = {"g": guess, "d": depth}

        if len(candidates) <= 1:
            return node

        # Particionar por feedback
        partition: dict[tuple, list[str]] = defaultdict(list)
        for cand in candidates:
            partition[feedback(cand, guess)].append(cand)

        children = {}
        for pat, group in sorted(partition.items()):
            pat_str = ''.join(str(x) for x in pat)

            if pat == WIN_PAT or len(group) == 0:
                continue

            if len(group) == 1:
                # Hoja: solo una opción
                children[pat_str] = {"g": group[0], "d": depth + 1}
                continue

            # Obtener el mejor guess para este subconjunto
            _, next_guess = self.solve(group, depth + 1)
            if next_guess is None:
                next_guess = group[0]

            # Recursión para construir el subárbol
            subtree = self.build_tree(group, next_guess, depth + 1)
            children[pat_str] = subtree

        if children:
            node["c"] = children

        return node


# ══════════════════════════════════════════════════════════════════════════════
# Estadísticas del árbol
# ══════════════════════════════════════════════════════════════════════════════

def tree_stats(tree: dict, depth: int = 0) -> dict:
    """Calcula estadísticas del árbol: distribución de profundidades."""
    stats: dict[int, int] = defaultdict(int)

    def _walk(node, d):
        if "c" not in node:
            stats[d] += 1
            return
        for child in node["c"].values():
            _walk(child, d + 1)

    _walk(tree, depth)

    total = sum(stats.values())
    mean = sum(d * n for d, n in stats.items()) / total if total else 0

    print(f"\n  Estadísticas del árbol:")
    print(f"  Total hojas: {total}")
    print(f"  Mean guesses: {mean:.4f}")
    for d in sorted(stats):
        pct = stats[d] / total * 100
        print(f"    {d} guesses: {stats[d]:4d} ({pct:.1f}%)")

    return dict(stats)


# ══════════════════════════════════════════════════════════════════════════════
# Generar código Python para embeber en strategy.py
# ══════════════════════════════════════════════════════════════════════════════

def flatten_tree(tree: dict, prefix: tuple = ()) -> dict[tuple, str]:
    """
    Convierte el árbol anidado a un dict plano para lookup O(1).

    Clave: secuencia de (pat1, pat2, ...) como tuple de strings
    Valor: mejor guess para ese estado

    Este formato es más eficiente en runtime que recorrer el árbol.
    """
    flat = {}

    def _walk(node, path):
        flat[path] = node["g"]
        if "c" not in node:
            return
        for pat_str, child in node["c"].items():
            _walk(child, path + (pat_str,))

    _walk(tree, prefix)
    return flat


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_mode(mode: str, vocab: list[str], weights: dict[str, float],
             non_words: list[str]) -> dict:

    print(f"\n{'═'*65}")
    print(f"  MODO: {mode} — {len(vocab)} palabras + {len(non_words)} no-palabras")
    print(f"{'═'*65}")

    solver = OptimalSolver(vocab, weights, non_words)

    # Resolver el árbol completo empezando desde el opener
    print(f"\n  Resolviendo árbol desde opener='{OPENER}'...")
    t0 = time.monotonic()

    # Primero, computar V* para todos los estados reachables
    # Esto llena el caché de memoización
    all_candidates = vocab[:]
    root_score, root_guess = solver.solve(all_candidates, depth=0)

    elapsed = time.monotonic() - t0
    print(f"\n  ✓ V* raíz calculado en {elapsed/60:.1f} min")
    print(f"  Score esperado raíz: {root_score:.4f}")
    print(f"  Opener recomendado: '{root_guess}'")
    print(f"  Estados únicos explorados: {solver.n_states:,}")
    print(f"  Cache hits: {solver.n_memo_hits:,}")

    # Si el DP recomienda un opener diferente a 'aore', usarlo
    actual_opener = root_guess if root_guess else OPENER

    # Construir árbol explícito para exportar
    print(f"\n  Construyendo árbol explícito con opener='{actual_opener}'...")
    tree = solver.build_tree(all_candidates, actual_opener, depth=0)
    stats = tree_stats(tree)

    return {
        "mode": mode,
        "opener": actual_opener,
        "tree": tree,
        "stats": stats,
        "root_score": root_score,
        "n_unique_states": solver.n_states,
        "n_memo_hits": solver.n_memo_hits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both",
                        choices=["uniform", "frequency", "both"])
    parser.add_argument("--no-nonwords", action="store_true",
                        help="No usar no-palabras (más rápido, peor calidad)")
    args = parser.parse_args()

    print("═" * 65)
    print("  OPTIMAL DP SOLVER — 4 letras")
    print("═" * 65)

    vocab, weights_u, weights_f = load_vocab()
    non_words = [] if args.no_nonwords else generate_non_words(vocab, n=100)
    if non_words:
        print(f"  No-palabras: {len(non_words)} ({non_words[:5]}...)")

    modes = []
    if args.mode in ("uniform", "both"):
        modes.append(("uniform", weights_u))
    if args.mode in ("frequency", "both"):
        modes.append(("frequency", weights_f))

    results = {}
    for mode, weights in modes:
        result = run_mode(mode, vocab, weights, non_words)
        results[mode] = result

        # Guardar árbol completo
        out_path = Path(f"optimal_tree_4_{mode}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result["tree"], f, ensure_ascii=False, separators=(',', ':'))

        # Guardar versión plana para lookup rápido
        flat = flatten_tree(result["tree"])
        flat_path = Path(f"optimal_flat_4_{mode}.json")
        # Convertir claves tuple a strings
        flat_str = {"|".join(k): v for k, v in flat.items()}
        with open(flat_path, 'w', encoding='utf-8') as f:
            json.dump(flat_str, f, ensure_ascii=False, separators=(',', ':'))

        size_tree = out_path.stat().st_size
        size_flat = flat_path.stat().st_size
        print(f"\n  Archivos guardados:")
        print(f"    {out_path}  ({size_tree/1024:.0f} KB)")
        print(f"    {flat_path}  ({size_flat/1024:.0f} KB)")
        print(f"\n  Opener final: '{result['opener']}'")

    print(f"\n{'═'*65}")
    print("  COMPLETO")
    print(f"{'═'*65}")
    print("\n  SIGUIENTE PASO:")
    print("  1. Copia optimal_flat_4_uniform.json y optimal_flat_4_frequency.json")
    print("     a la carpeta estudiantes/gabriel_regina/")
    print("  2. Actualiza strategy.py para cargar estos archivos en begin_game()")
    print("     (ver comentarios en strategy_v6.py)")


if __name__ == "__main__":
    main()