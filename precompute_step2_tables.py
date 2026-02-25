"""
PASO 2: Computar tablas T2 y T3 con DP exacto.

Prerrequisito: tener los openers del paso 1 en precomputed_openers.txt
Edita las variables OPENERS_* abajo con esos resultados.

Corre desde:  cd ~/ia/wordle/torneo_wordle
Comando:      python3 precompute_step2_tables.py

Tiempo estimado (en tu máquina, un núcleo):
  T2: ~30-90 min (dominado por 6-letras: 450 patrones × pool completo)
  T3: ~1-3 horas (muchos subproblemas pero candidatos pequeños)

El script guarda progreso incremental por escenario, así que si se interrumpe
puedes reiniciarlo — detecta archivos .json ya existentes y los salta.

Output: precomputed_tables_raw.py  (importable en strategy.py)
"""

import csv
import itertools
import json
import math
import re
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


# ══════════════════════════════════════════════════════════════════════════════
# *** EDITA CON LOS RESULTADOS DEL PASO 1 ***
# ══════════════════════════════════════════════════════════════════════════════
OPENERS_UNIFORM = {4: 'aore', 5: 'careo', 6: 'carieo'}
OPENERS_FREQ    = {4: 'aore', 5: 'careo', 6: 'carieo'}
# Nota: los de frequency están MAL en la primera corrida (usaban naive probs).
# Al reejecutar step1 con _sigmoid_weights, actualiza estos valores.


# ══════════════════════════════════════════════════════════════════════════════
# Carga de vocabulario — replica exactamente lexicon.py
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


def load_vocab(wl: int) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """
    Devuelve (words, weights_uniform, weights_frequency) exactamente
    como los genera lexicon.py para el torneo.

    weights_uniform:  1/N para todas las palabras
    weights_frequency: _sigmoid_weights(counts, steepness=1.5)
    """
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
# No-palabras
# ══════════════════════════════════════════════════════════════════════════════

def generate_non_words(vocab: list[str], wl: int, n: int = 50) -> list[str]:
    vocab_set = set(vocab)
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
    pool = top[:min(wl + 5, len(top))]

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
# DP: expected score exacto
# ══════════════════════════════════════════════════════════════════════════════

def f_hat(n: int) -> float:
    if n <= 1: return 1.0
    if n == 2: return 1.5
    if n == 3: return 2.0
    return max(1.0, math.log2(n) * 0.5 + 0.8)


def normalize(candidates: list[str], base_w: dict[str, float]) -> dict[str, float]:
    raw = {w: base_w.get(w, 1e-10) for w in candidates}
    total = sum(raw.values())
    if total == 0:
        return {w: 1.0 / len(candidates) for w in candidates}
    return {w: v / total for w, v in raw.items()}


def expected_score(guess_word: str,
                   candidates: list[str],
                   weights: dict[str, float],
                   wl: int) -> float:
    """DP exacto de un paso: Score(g) = Σ_f p(f|g) × cost(f)"""
    win_pat = tuple([2] * wl)
    partition = defaultdict(list)
    for w in candidates:
        partition[feedback(w, guess_word)].append(w)

    total_w = sum(weights.get(w, 0.0) for w in candidates)
    if total_w == 0:
        total_w = 1.0

    score = 0.0
    for pat, group in partition.items():
        p_f = sum(weights.get(w, 0.0) for w in group) / total_w
        score += p_f * (1.0 if pat == win_pat else 1.0 + f_hat(len(group)))
    return score


def best_guess_in_pool(candidates: list[str],
                       weights: dict[str, float],
                       pool: list[str],
                       wl: int) -> str:
    """Devuelve el guess de menor expected_score dentro del pool dado."""
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return max(candidates, key=lambda w: weights.get(w, 0.0))

    w = normalize(candidates, weights)
    best, best_s = pool[0], float('inf')
    for g in pool:
        s = expected_score(g, candidates, w, wl)
        if s < best_s:
            best_s = s
            best = g
    return best


# ══════════════════════════════════════════════════════════════════════════════
# T2: opener_feedback_pattern → best_second_guess
# ══════════════════════════════════════════════════════════════════════════════

def compute_t2(opener: str,
               vocab: list[str],
               base_w: dict[str, float],
               non_words: list[str],
               wl: int,
               label: str) -> dict[str, str]:
    """
    Para cada patrón de feedback tras el opener, encuentra el mejor T2.
    Pool de evaluación: vocabulario completo + no-palabras.
    Esto es costoso (~30 min para 6-letras) pero se hace UNA vez offline.
    """
    print(f"\n  T2 {label} opener='{opener}'")
    win_pat = tuple([2] * wl)
    full_pool = vocab + non_words  # evaluamos TODO el vocab, sin shortcuts

    # Agrupar candidatos por patrón T1
    t1_groups: dict[tuple, list[str]] = defaultdict(list)
    for w in vocab:
        t1_groups[feedback(w, opener)].append(w)

    print(f"  {len(t1_groups)} patrones T1, "
          f"pool={len(full_pool)} guesses × vocab={len(vocab)}")

    t2 = {}
    t0 = time.monotonic()
    n = len(t1_groups)

    for i, (pat, cands) in enumerate(sorted(t1_groups.items())):
        if pat == win_pat:
            t2[''.join(str(x) for x in pat)] = opener
            continue

        # Para grupos pequeños, evaluamos solo los candidatos (idéntico al full)
        # Para grupos grandes, necesitamos el pool completo para encontrar
        # posibles guesses fuera del conjunto de candidatos que den más info
        pool = full_pool if len(cands) > 5 else cands
        w = normalize(cands, base_w)
        best = best_guess_in_pool(cands, w, pool, wl)
        t2[''.join(str(x) for x in pat)] = best

        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    [{i+1}/{n}] ~{eta:.0f}s restantes — "
                  f"último pat={pat} cands={len(cands)} → '{best}'")

    print(f"  ✓ T2 {label}: {len(t2)} entradas en "
          f"{(time.monotonic()-t0)/60:.1f}min")
    return t2


# ══════════════════════════════════════════════════════════════════════════════
# T3: (pat_t1, guess_t2, pat_t2) → best_third_guess
# ══════════════════════════════════════════════════════════════════════════════

def compute_t3(opener: str,
               t2_table: dict[str, str],
               vocab: list[str],
               base_w: dict[str, float],
               non_words: list[str],
               wl: int,
               label: str) -> dict[str, str]:
    """
    Para cada rama que llega a T3 con >2 candidatos, encuentra el mejor T3.

    Para T3, el número de candidatos típicamente es pequeño (2-20).
    Evaluamos solo los candidatos (no el vocab completo) porque:
    1. Con <20 candidatos, la ganancia de buscar fuera es mínima
    2. Reduce el cómputo de horas a minutos
    Excepción: si hay >20 candidatos en T3, añadimos no-palabras.
    """
    print(f"\n  T3 {label}")
    win_pat = tuple([2] * wl)

    # Reconstruir grupos T1
    t1_groups: dict[tuple, list[str]] = defaultdict(list)
    for w in vocab:
        t1_groups[feedback(w, opener)].append(w)

    t3 = {}
    total_nodes = 0
    t0 = time.monotonic()

    for pat1, cands1 in sorted(t1_groups.items()):
        if pat1 == win_pat or len(cands1) <= 2:
            continue

        pat1_str = ''.join(str(x) for x in pat1)
        guess2 = t2_table.get(pat1_str)
        if guess2 is None:
            continue

        # Agrupar por patrón T2
        t2_groups: dict[tuple, list[str]] = defaultdict(list)
        for w in cands1:
            t2_groups[feedback(w, guess2)].append(w)

        for pat2, cands2 in t2_groups.items():
            if pat2 == win_pat or len(cands2) <= 2:
                continue

            key = f"{pat1_str}|{guess2}|{''.join(str(x) for x in pat2)}"
            w = normalize(cands2, base_w)

            # Pool para T3: candidatos + no-palabras si hay muchos
            pool = cands2 + non_words[:20] if len(cands2) > 15 else cands2
            best = best_guess_in_pool(cands2, w, pool, wl)
            t3[key] = best
            total_nodes += 1

    print(f"  ✓ T3 {label}: {len(t3)} entradas en "
          f"{(time.monotonic()-t0)/60:.1f}min")
    return t3


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("PASO 2: Tablas T2 y T3")
    print("=" * 65)

    all_data = {}

    # Cargar datos parciales existentes (por si el script se interrumpió)
    for wl in [4, 5, 6]:
        for mode in ['uniform', 'frequency']:
            key = f"{wl}_{mode}"
            partial = Path(f"precomputed_{key}.json")
            if partial.exists():
                with open(partial) as f:
                    all_data[key] = json.load(f)
                print(f"  ✓ Cargado {partial} (ya existía)")

    for wl in [4, 5, 6]:
        print(f"\n{'='*65}\nLONGITUD {wl} LETRAS\n{'='*65}")
        vocab, weights_u, weights_f = load_vocab(wl)
        non_words = generate_non_words(vocab, wl, n=50)

        for mode, base_w, opener in [
            ("uniform",   weights_u, OPENERS_UNIFORM[wl]),
            ("frequency", weights_f, OPENERS_FREQ[wl]),
        ]:
            key = f"{wl}_{mode}"
            if key in all_data:
                print(f"  Saltando {key} (ya computado)")
                continue

            label = f"{wl}-{mode}"
            t2 = compute_t2(opener, vocab, base_w, non_words, wl, label)
            t3 = compute_t3(opener, t2, vocab, base_w, non_words, wl, label)

            all_data[key] = {
                'opener': opener,
                't2': t2,
                't3': t3,
            }

            # Guardar parcialmente por si se interrumpe
            with open(f"precomputed_{key}.json", 'w', encoding='utf-8') as f:
                json.dump(all_data[key], f, ensure_ascii=False)
            print(f"  Guardado precomputed_{key}.json")

    # Generar archivo Python listo para usar en strategy.py
    _generate_py(all_data)
    print("\n✓ Listo. Siguiente paso: python3 build_strategy.py")


def _generate_py(all_data: dict):
    """Genera precomputed_tables_raw.py con los dicts hardcodeados."""
    lines = [
        '"""',
        'Tablas precomputadas — generado por precompute_step2_tables.py',
        'NO editar manualmente.',
        '"""\n',
        '# fmt: off',
        '',
    ]

    openers_u = {int(k.split('_')[0]): v['opener']
                 for k, v in all_data.items() if k.endswith('_uniform')}
    openers_f = {int(k.split('_')[0]): v['opener']
                 for k, v in all_data.items() if k.endswith('_frequency')}

    lines.append(f"OPENERS_UNIFORM = {openers_u}")
    lines.append(f"OPENERS_FREQ    = {openers_f}")
    lines.append("")

    for key, data in sorted(all_data.items()):
        wl, mode = key.split('_', 1)
        lines.append(f"# {wl}-letras {mode}: "
                     f"{len(data['t2'])} entradas T2, "
                     f"{len(data['t3'])} entradas T3")
        lines.append(f"_T2_{wl}_{mode.upper()} = {repr(data['t2'])}")
        lines.append(f"_T3_{wl}_{mode.upper()} = {repr(data['t3'])}")
        lines.append("")

    lines += [
        "T2_TABLES = {",
    ]
    for wl in [4, 5, 6]:
        lines.append(f"    {wl}: {{")
        lines.append(f"        'uniform':   _T2_{wl}_UNIFORM,")
        lines.append(f"        'frequency': _T2_{wl}_FREQUENCY,")
        lines.append(f"    }},")
    lines.append("}")
    lines.append("")
    lines += [
        "T3_TABLES = {",
    ]
    for wl in [4, 5, 6]:
        lines.append(f"    {wl}: {{")
        lines.append(f"        'uniform':   _T3_{wl}_UNIFORM,")
        lines.append(f"        'frequency': _T3_{wl}_FREQUENCY,")
        lines.append(f"    }},")
    lines.append("}")

    with open("precomputed_tables_raw.py", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("\n✓ precomputed_tables_raw.py generado")
    _print_sizes(all_data)


def _print_sizes(all_data: dict):
    total_bytes = 0
    for key, data in sorted(all_data.items()):
        t2_n = len(data['t2'])
        t3_n = len(data['t3'])
        # Estimación de bytes: cada entrada ~40 chars clave + 10 valor
        est = t2_n * 50 + t3_n * 60
        total_bytes += est
        print(f"  {key:20s}: T2={t2_n:4d}  T3={t3_n:5d}  ~{est//1024}KB")
    print(f"  {'TOTAL':20s}: ~{total_bytes//1024}KB estimado en strategy.py")


if __name__ == "__main__":
    main()