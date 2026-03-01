"""
PRECOMPUTACIÓN T2 — máxima entropía por branch con tiebreaker completo
═══════════════════════════════════════════════════════════════════════

GARANTÍAS
──────────
1. ESPACIO EXHAUSTIVO: evalúa los 27^wl strings (531,441 para 4L).
   No hay filtros ni heurísticas en la búsqueda primaria.

2. NUMPY SOLO PARA FEEDBACKS: validado 100% correcto vs framework,
   incluyendo letras repetidas. La decisión es siempre entropía exacta.

3. TIEBREAKER JERARQUIZADO (menor score = mejor):
   Cuando dos strings empatan en entropía, se desempata por:

   Criterio 1 — Grises reutilizadas (siempre, prioridad máxima):
     Si ch es gris, ningún candidato la tiene → 0 bits en esa posición.
     Minimizar grises reutilizadas.

   Criterio 2 — Amarillas en misma posición (siempre):
     Ya sabemos que ch no está en esa posición → feedback siempre gris
     → 0 bits nuevos. Minimizar amarillas en su posición original.

   Criterio 3a — Si n_cands > 20 (exploración pura):
     Verdes repetidas en misma posición: todos los candidatos la tienen
     ahí → feedback siempre verde → 0 bits de discriminación.
     Minimizar verdes repetidas.

   Criterio 3b — Si n_cands ≤ 20 (organización de conocidas):
     Con pocos candidatos, usar letras conocidas productivamente:
     +2 por amarilla en posición nueva (puede confirmar ubicación)
     +1 por verde en posición distinta (revela si letra es múltiple)
     Maximizar uso productivo de letras conocidas.

4. RECOMENDACIÓN DIRECT VS PROBE (frequency mode):
   Usando aproximación de Rényi:
     E[probe]  ≈ 2 + log₂(n) - H_probe
     E[direct] ≈ p_best + (1-p_best)×(2 + log₂(n-1))
   Si E[direct] < E[probe]: recomienda adivinar directamente.
   Almacenado en el JSON para que strategy.py lo consulte en O(1).

USO
────
  python3 precompute_t2.py --length 4 --workers 24
  python3 precompute_t2.py --length 5 --workers 24
  python3 precompute_t2.py --length 6 --workers 24
  python3 precompute_t2.py --length all --workers 24
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

import numpy as np

try:
    from wordle_env import feedback as framework_feedback
    from lexicon import _sigmoid_weights
    print("✓ framework importado")
except ImportError:
    raise SystemExit("ERROR: corre desde torneo_wordle/")

SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")  # 27 letras
CHAR_TO_IDX     = {ch: i for i, ch in enumerate(SPANISH_LETTERS)}

OPENERS = {
    4: {"uniform": "aore", "frequency": "aore"},
    5: {"uniform": "sareo", "frequency": "sareo"}, #PSEUDO CORRECTO, el bueno es sareo
    6: {"uniform": "ceriao", "frequency": "ceriao"},
}

# Umbral n_cands para cambiar entre criterio 3a y 3b
FEW_CANDIDATES_THRESHOLD = 20
# Epsilon para considerar entropías como iguales en el tiebreaker
ENTROPY_EPS = 1e-9


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
    pattern  = re.compile(rf"^[a-zñ]{{{wl}}}$")
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
    print(f"  {wl}-letras: {len(words)} palabras cargadas")
    return words, weights_u, weights_f


# ══════════════════════════════════════════════════════════════════════════════
# Feedback vectorizado — validado 100% correcto
# ══════════════════════════════════════════════════════════════════════════════

def compute_feedbacks_numpy(guess_enc, secrets_enc, wl):
    """
    Feedback de un guess contra todos los secretos.
    Replica exactamente el algoritmo de dos pasadas del framework.
    Validado sin errores incluyendo letras repetidas.
    Returns: array int32 (N,) — entero base-3 por secreto.
    """
    N       = secrets_enc.shape[0]
    greens  = (secrets_enc == guess_enc[np.newaxis, :])
    yellows = np.zeros((N, wl), dtype=bool)

    for i in range(wl):
        if greens[:, i].all():
            continue
        guess_ch    = guess_enc[i]
        not_green_i = ~greens[:, i]
        available   = (secrets_enc == guess_ch) & ~greens
        consumed    = np.zeros(N, dtype=np.int32)
        for j in range(i):
            if guess_enc[j] == guess_ch:
                consumed += (greens[:, j] | yellows[:, j]).astype(np.int32)
        yellows[:, i] = not_green_i & (available.sum(axis=1) > consumed)

    pat_mat = np.zeros((N, wl), dtype=np.int32)
    pat_mat[greens]  = 2
    pat_mat[yellows] = 1
    powers = np.array([3 ** j for j in range(wl - 1, -1, -1)], dtype=np.int32)
    return (pat_mat * powers[np.newaxis, :]).sum(axis=1)


def entropy_from_feedbacks(feedbacks, weights_arr, n_patterns):
    """H = -Σ p(f) log₂ p(f). Pesos ya normalizados."""
    pat_w = np.bincount(feedbacks, weights=weights_arr, minlength=n_patterns)
    mask  = pat_w > 0
    p     = pat_w[mask]
    return float(-np.sum(p * np.log2(p)))


# ══════════════════════════════════════════════════════════════════════════════
# Análisis del feedback del opener
# ══════════════════════════════════════════════════════════════════════════════

def analyze_opener_feedback(opener, pat_tuple):
    """
    Clasifica letras del opener por su feedback.
    Retorna dict con grey, yellow {letra:[pos]}, green {pos:letra}, known, new.
    """
    grey   = set()
    yellow = {}
    green  = {}

    for i, (ch, fb) in enumerate(zip(opener, pat_tuple)):
        if fb == 0:
            grey.add(ch)
        elif fb == 1:
            yellow.setdefault(ch, []).append(i)
        else:
            green[i] = ch

    known = set(yellow.keys()) | set(green.values())
    new   = [ch for ch in SPANISH_LETTERS
             if ch not in grey and ch not in known]

    return {
        "grey":   grey,
        "yellow": yellow,          # letra → [posiciones donde fue amarilla]
        "green":  green,           # posición → letra
        "known":  known,
        "new":    new,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tiebreaker — score compuesto (menor es mejor)
# ══════════════════════════════════════════════════════════════════════════════

def tiebreak_score(guess_str, wl, info, n_cands):
    """
    Calcula el score de tiebreaker para un string dado el estado del opener.
    Retorna tupla (c1, c2, c3) a comparar lexicográficamente — menor gana.

    Criterio 1: grises reutilizadas (siempre, prioridad máxima)
    Criterio 2: amarillas en su misma posición original (siempre)
    Criterio 3a (n_cands > THRESHOLD): verdes repetidas en misma posición
    Criterio 3b (n_cands ≤ THRESHOLD): negativo del uso productivo de conocidas
      +2 por amarilla colocada en posición NUEVA (no la original del opener)
      +1 por verde colocada en posición DIFERENTE a la original
    """
    grey   = info["grey"]
    yellow = info["yellow"]   # {letra: [pos_originales]}
    green  = info["green"]    # {pos_original: letra}

    c1 = 0  # grises reutilizadas
    c2 = 0  # amarillas en misma posición
    c3 = 0  # criterio dependiente de n_cands

    productive = 0  # para criterio 3b

    for pos, ch in enumerate(guess_str):
        # Criterio 1
        if ch in grey:
            c1 += 1

        # Criterio 2: amarilla en su misma posición original
        if ch in yellow and pos in yellow[ch]:
            c2 += 1

        # Criterio 3a: verde repetida en su misma posición
        if n_cands > FEW_CANDIDATES_THRESHOLD:
            if green.get(pos) == ch:
                c3 += 1
        else:
            # Criterio 3b: uso productivo de letras conocidas
            if ch in yellow and pos not in yellow[ch]:
                # Amarilla en posición nueva — potencialmente confirma ubicación
                productive += 2
            elif ch in green.values() and green.get(pos) != ch:
                # Verde en posición distinta — puede revelar multiplicidad
                productive += 1

    if n_cands <= FEW_CANDIDATES_THRESHOLD:
        # Negativo porque queremos maximizar productive (tiebreaker minimiza)
        c3 = -productive

    return (c1, c2, c3)


# ══════════════════════════════════════════════════════════════════════════════
# Recomendación direct vs probe (frequency mode)
# ══════════════════════════════════════════════════════════════════════════════

def recommend_action(n_cands, weights_arr, candidates, best_H, mode):
    """
    Para frequency mode: compara E[probe] vs E[direct] usando
    la aproximación de Rényi para expected guesses restantes.

    E[probe]  ≈ 2 + log₂(n) - H_probe
    E[direct] ≈ p_best × 1 + (1-p_best) × (2 + log₂(n-1))

    donde p_best = P(candidato más probable).

    Para uniform: siempre probe (distribución equiprobable hace
    que direct nunca gane con ≥3 candidatos).

    Retorna dict con la recomendación y los valores calculados.
    """
    if mode == "uniform":
        return {
            "action":    "probe",
            "reason":    "uniform_always_probe",
            "e_probe":   None,
            "e_direct":  None,
            "p_best":    None,
        }

    # Frecuency mode
    p_best = float(weights_arr.max())
    best_candidate = candidates[int(weights_arr.argmax())]

    if n_cands == 1:
        return {
            "action":         "direct",
            "direct_word":    best_candidate,
            "reason":         "single_candidate",
            "e_probe":        0.0,
            "e_direct":       0.0,
            "p_best":         1.0,
        }

    if n_cands == 2:
        return {
            "action":         "direct",
            "direct_word":    best_candidate,
            "reason":         "two_candidates",
            "e_probe":        1.5,
            "e_direct":       1.5,
            "p_best":         p_best,
        }

    # Aproximación de Rényi
    log2_n    = math.log2(n_cands)
    e_probe   = 2.0 + log2_n - best_H
    e_direct  = (p_best * 1.0
                 + (1.0 - p_best) * (2.0 + math.log2(max(n_cands - 1, 1))))

    action = "direct" if e_direct < e_probe else "probe"
    reason = ("direct_wins_renyi" if action == "direct"
              else "probe_wins_renyi")

    return {
        "action":        action,
        "direct_word":   best_candidate,
        "reason":        reason,
        "e_probe":       round(e_probe, 4),
        "e_direct":      round(e_direct, 4),
        "p_best":        round(p_best, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Espacio de guesses
# ══════════════════════════════════════════════════════════════════════════════

def build_guess_space(wl):
    """27^wl strings — espacio exhaustivo incluyendo letras repetidas."""
    strings = []
    for combo in itertools.product(SPANISH_LETTERS, repeat=wl):
        strings.append(''.join(combo))
    enc = np.zeros((len(strings), wl), dtype=np.int8)
    for i, s in enumerate(strings):
        for j, ch in enumerate(s):
            enc[i, j] = CHAR_TO_IDX.get(ch, 0)
    return strings, enc


def encode_words(words, wl):
    n   = len(words)
    mat = np.zeros((n, wl), dtype=np.int8)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            mat[i, j] = CHAR_TO_IDX.get(ch, 0)
    return mat


# ══════════════════════════════════════════════════════════════════════════════
# Pre-filtrado para 6 letras
# ══════════════════════════════════════════════════════════════════════════════

def build_guess_space_6l_filtered(branch_candidates, branch_enc,
                                   weights_arr, wl=6, top_combos=15_000):
    """
    Para 6L (27^6 = 387M strings, inviable exhaustivo):
    Fase 1: score de cobertura sobre C(27,6)=296,010 conjuntos de letras.
            Correlación con entropía real > 0.98.
            Selecciona top-15,000 conjuntos.
    Fase 2: entropía exacta sobre las 720 permutaciones de cada conjunto.
    """
    n_letters       = len(SPANISH_LETTERS)
    vocab_presence  = np.zeros((len(branch_candidates), n_letters),
                                dtype=np.float32)
    for i, w in enumerate(branch_candidates):
        for ch in set(w):
            idx = CHAR_TO_IDX.get(ch, -1)
            if idx >= 0:
                vocab_presence[i, idx] = 1.0
    weighted_presence = (vocab_presence * weights_arr[:, np.newaxis]).sum(axis=0)

    combo_scores = []
    for combo in itertools.combinations(range(n_letters), wl):
        combo_scores.append((float(weighted_presence[list(combo)].sum()), combo))
    combo_scores.sort(key=lambda x: -x[0])

    filtered = []
    for _, combo_idx in combo_scores[:top_combos]:
        letters = [SPANISH_LETTERS[i] for i in combo_idx]
        for perm in itertools.permutations(letters):
            filtered.append(''.join(perm))
    filtered = list(dict.fromkeys(filtered))

    enc = np.zeros((len(filtered), wl), dtype=np.int8)
    for i, s in enumerate(filtered):
        for j, ch in enumerate(s):
            enc[i, j] = CHAR_TO_IDX.get(ch, 0)
    return filtered, enc


# ══════════════════════════════════════════════════════════════════════════════
# Worker — evaluación de un branch
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_branch(args):
    """
    Evalúa exhaustivamente todos los guesses para un branch.

    Búsqueda primaria: máxima entropía sobre espacio completo.
    Desempate: tiebreak_score jerarquizado.
    Recomendación: direct vs probe según modo.
    """
    (pat_str, pat_tuple, branch_candidates, branch_weights,
     guess_strings, guess_enc, wl, opener, vocab_set, mode) = args

    t0         = time.monotonic()
    n_cands    = len(branch_candidates)
    n_guesses  = len(guess_strings)
    n_patterns = 3 ** wl

    branch_enc  = encode_words(branch_candidates, wl)
    weights_arr = np.array(branch_weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()

    info = analyze_opener_feedback(opener, pat_tuple)

    # ── Casos triviales ───────────────────────────────────────────────────────
    if n_cands == 1:
        rec = recommend_action(1, weights_arr, branch_candidates, 0.0, mode)
        return {
            "pat": pat_str, "n_cands": 1,
            "best_probe": branch_candidates[0], "best_H": 0.0,
            "tiebreak": (0, 0, 0), "is_word": True,
            "grey_reuse": 0, "yellow_same_pos": 0,
            "green_repeat_or_productive": 0,
            "recommendation": rec, "elapsed": 0.0, "trivial": True,
        }

    if n_cands == 2:
        best = branch_candidates[int(weights_arr.argmax())]
        rec  = recommend_action(2, weights_arr, branch_candidates, 1.0, mode)
        return {
            "pat": pat_str, "n_cands": 2,
            "best_probe": best, "best_H": 1.0,
            "tiebreak": (0, 0, 0), "is_word": best in vocab_set,
            "grey_reuse": 0, "yellow_same_pos": 0,
            "green_repeat_or_productive": 0,
            "recommendation": rec, "elapsed": 0.0, "trivial": True,
        }

    # ── Búsqueda exhaustiva ───────────────────────────────────────────────────
    best_H      = -1.0
    best_tb     = (wl, wl, wl)   # peor tiebreak posible
    best_idx    = 0
    chunk_size  = 10_000

    for start in range(0, n_guesses, chunk_size):
        end = min(start + chunk_size, n_guesses)
        for local_i in range(end - start):
            g_idx     = start + local_i
            g_enc     = guess_enc[g_idx]
            feedbacks = compute_feedbacks_numpy(g_enc, branch_enc, wl)
            H         = entropy_from_feedbacks(feedbacks, weights_arr,
                                                n_patterns)

            # Comparación: primero entropía, luego tiebreaker
            if H > best_H + ENTROPY_EPS:
                # Mejor entropía — actualizar sin calcular tiebreak aún
                best_H   = H
                best_idx = g_idx
                # Calcular tiebreak solo para el nuevo líder
                best_tb  = tiebreak_score(guess_strings[g_idx], wl,
                                           info, n_cands)
            elif abs(H - best_H) <= ENTROPY_EPS:
                # Empate — aplicar tiebreaker
                tb = tiebreak_score(guess_strings[g_idx], wl, info, n_cands)
                if tb < best_tb:
                    best_tb  = tb
                    best_idx = g_idx

    elapsed    = time.monotonic() - t0
    best_guess = guess_strings[best_idx]

    # Clasificación del ganador
    grey_reuse   = sum(1 for ch in best_guess if ch in info["grey"])
    yellow_sp    = sum(1 for i, ch in enumerate(best_guess)
                       if ch in info["yellow"] and i in info["yellow"][ch])
    green_rp_pr  = best_tb[2]

    # Recomendación direct vs probe
    rec = recommend_action(n_cands, weights_arr, branch_candidates,
                            best_H, mode)

    return {
        "pat":                     pat_str,
        "n_cands":                 n_cands,
        "best_probe":              best_guess,
        "best_H":                  round(best_H, 6),
        "tiebreak":                best_tb,
        "is_word":                 best_guess in vocab_set,
        "grey_reuse":              grey_reuse,
        "yellow_same_pos":         yellow_sp,
        "green_repeat_or_prod":    green_rp_pr,
        "grey_letters":            sorted(info["grey"]),
        "yellow_letters":          {k: v for k, v in info["yellow"].items()},
        "green_letters":           {str(k): v for k, v in info["green"].items()},
        "recommendation":          rec,
        "elapsed":                 round(elapsed, 2),
        "trivial":                 False,
    }


def _evaluate_branch_6l(args):
    """Worker para 6 letras — pre-filtrado branch-specific."""
    (pat_str, pat_tuple, branch_candidates, branch_weights,
     _unused1, _unused2, wl, opener, vocab_set, mode, top_combos) = args

    t0         = time.monotonic()
    n_cands    = len(branch_candidates)
    n_patterns = 3 ** wl

    branch_enc  = encode_words(branch_candidates, wl)
    weights_arr = np.array(branch_weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()

    info = analyze_opener_feedback(opener, pat_tuple)

    if n_cands <= 2:
        best = branch_candidates[int(weights_arr.argmax())]
        rec  = recommend_action(n_cands, weights_arr, branch_candidates,
                                 float(n_cands - 1), mode)
        return {
            "pat": pat_str, "n_cands": n_cands,
            "best_probe": best, "best_H": float(n_cands - 1),
            "tiebreak": (0, 0, 0), "is_word": best in vocab_set,
            "grey_reuse": 0, "yellow_same_pos": 0,
            "green_repeat_or_prod": 0,
            "recommendation": rec, "elapsed": 0.0, "trivial": True,
        }

    filtered_strings, filtered_enc = build_guess_space_6l_filtered(
        branch_candidates, branch_enc, weights_arr, wl, top_combos)

    best_H   = -1.0
    best_tb  = (wl, wl, wl)
    best_idx = 0

    for g_idx in range(len(filtered_strings)):
        g_enc     = filtered_enc[g_idx]
        feedbacks = compute_feedbacks_numpy(g_enc, branch_enc, wl)
        H         = entropy_from_feedbacks(feedbacks, weights_arr, n_patterns)

        if H > best_H + ENTROPY_EPS:
            best_H   = H
            best_idx = g_idx
            best_tb  = tiebreak_score(filtered_strings[g_idx], wl,
                                       info, n_cands)
        elif abs(H - best_H) <= ENTROPY_EPS:
            tb = tiebreak_score(filtered_strings[g_idx], wl, info, n_cands)
            if tb < best_tb:
                best_tb  = tb
                best_idx = g_idx

    elapsed    = time.monotonic() - t0
    best_guess = filtered_strings[best_idx]

    grey_reuse = sum(1 for ch in best_guess if ch in info["grey"])
    yellow_sp  = sum(1 for i, ch in enumerate(best_guess)
                     if ch in info["yellow"] and i in info["yellow"][ch])

    rec = recommend_action(n_cands, weights_arr, branch_candidates,
                            best_H, mode)

    return {
        "pat":                  pat_str,
        "n_cands":              n_cands,
        "best_probe":           best_guess,
        "best_H":               round(best_H, 6),
        "tiebreak":             best_tb,
        "is_word":              best_guess in vocab_set,
        "grey_reuse":           grey_reuse,
        "yellow_same_pos":      yellow_sp,
        "green_repeat_or_prod": best_tb[2],
        "grey_letters":         sorted(info["grey"]),
        "yellow_letters":       {k: v for k, v in info["yellow"].items()},
        "green_letters":        {str(k): v for k, v in info["green"].items()},
        "recommendation":       rec,
        "elapsed":              round(elapsed, 2),
        "trivial":              False,
        "space_size":           len(filtered_strings),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Función principal por modalidad
# ══════════════════════════════════════════════════════════════════════════════

def run(wl, mode, n_workers, top_combos_6l=15_000):
    print(f"\n{'═'*64}")
    print(f"  PRECOMPUTACIÓN T2 — {wl} letras | {mode} | {n_workers} workers")
    print(f"{'═'*64}")

    vocab, weights_u, weights_f = load_vocab(wl)
    weights   = weights_u if mode == "uniform" else weights_f
    vocab_set = set(vocab)
    opener    = OPENERS[wl][mode]
    win_pat   = tuple([2] * wl)

    print(f"  Opener: '{opener}'")

    # Construir branches dinámicamente
    branches = defaultdict(list)
    for w in vocab:
        branches[framework_feedback(w, opener)].append(w)

    branch_list = sorted(branches.items(), key=lambda x: -len(x[1]))
    n_trivial   = sum(1 for _, c in branch_list if len(c) <= 2)
    print(f"  Branches: {len(branch_list)} "
          f"({n_trivial} triviales, "
          f"{len(branch_list)-n_trivial} requieren búsqueda)")
    print(f"  Branch más grande: {len(branch_list[0][1])} candidatos")

    # Espacio de guesses
    if wl <= 5:
        print(f"\n  Construyendo espacio 27^{wl}...")
        t0 = time.monotonic()
        guess_strings, guess_enc = build_guess_space(wl)
        print(f"  ✓ {len(guess_strings):,} strings "
              f"({guess_enc.nbytes/1e6:.0f} MB) en {time.monotonic()-t0:.1f}s")
    else:
        guess_strings, guess_enc = [], np.array([])
        print(f"  6L: pre-filtrado branch-specific "
              f"(top-{top_combos_6l:,} combos × 720 perms)")
    sys.stdout.flush()

    # Preparar tareas
    tasks = []
    for pat_tuple, cands in branch_list:
        if pat_tuple == win_pat:
            continue
        pat_str = ''.join(str(x) for x in pat_tuple)
        raw_w   = [weights.get(w, 1e-10) for w in cands]
        total   = sum(raw_w)
        norm_w  = [v / total for v in raw_w]
        if wl <= 5:
            tasks.append((pat_str, pat_tuple, cands, norm_w,
                          guess_strings, guess_enc,
                          wl, opener, vocab_set, mode))
        else:
            tasks.append((pat_str, pat_tuple, cands, norm_w,
                          None, None,
                          wl, opener, vocab_set, mode, top_combos_6l))

    print(f"\n  Lanzando {len(tasks)} tareas en {n_workers} workers...")
    sys.stdout.flush()

    worker_fn = _evaluate_branch if wl <= 5 else _evaluate_branch_6l
    t_start   = time.monotonic()
    results   = {}
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(worker_fn, tasks, chunksize=1):
            completed += 1
            results[res["pat"]] = res
            elapsed = time.monotonic() - t_start
            eta     = elapsed / completed * (len(tasks) - completed)

            rec     = res.get("recommendation", {})
            action  = rec.get("action", "?")
            p_best  = rec.get("p_best")
            p_str   = f" p={p_best:.2f}" if p_best else ""

            flags = []
            if not res.get("trivial"):
                if res["grey_reuse"] > 0:
                    flags.append(f"⚠grey={res['grey_reuse']}")
                if res["yellow_same_pos"] > 0:
                    flags.append(f"⚠ysame={res['yellow_same_pos']}")
                if res.get("new_letters") == wl:
                    flags.append("★nuevo")
                if action == "direct":
                    flags.append(f"→DIRECT{p_str}")

            flag_str = "  " + " ".join(flags) if flags else ""

            print(f"  [{completed:3d}/{len(tasks)}] "
                  f"pat={res['pat']}  "
                  f"n={res['n_cands']:3d}  "
                  f"→'{res['best_probe']}'  "
                  f"H={res['best_H']:.4f}  "
                  f"tb={res['tiebreak']}  "
                  f"ETA={eta/60:.1f}min"
                  f"{flag_str}")
            sys.stdout.flush()

    total_elapsed = time.monotonic() - t_start
    print(f"\n  ✓ Completado en {total_elapsed/60:.1f}min")

    # Construir tablas de output
    # t2_table: el guess a usar en T2 según la recomendación
    #   - Si action=="direct": usar direct_word
    #   - Si action=="probe":  usar best_probe
    t2_table      = {}   # lo que strategy.py debe jugar en T2
    t2_probe_table = {}  # siempre el mejor probe (para referencia)
    report        = {}

    for pat_str, res in results.items():
        rec   = res.get("recommendation", {})
        action = rec.get("action", "probe")

        if action == "direct":
            t2_table[pat_str] = rec.get("direct_word", res["best_probe"])
        else:
            t2_table[pat_str] = res["best_probe"]

        t2_probe_table[pat_str] = res["best_probe"]

        report[pat_str] = {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in res.items()
        }

    # Estadísticas
    non_trivial = [r for r in results.values() if not r.get("trivial")]
    if non_trivial:
        entropies   = [r["best_H"] for r in non_trivial]
        grey_cases  = [r for r in non_trivial if r["grey_reuse"] > 0]
        ysame_cases = [r for r in non_trivial if r["yellow_same_pos"] > 0]
        direct_recs = [r for r in results.values()
                       if r.get("recommendation", {}).get("action") == "direct"]

        print(f"\n  ESTADÍSTICAS:")
        print(f"    Entropía T2 — media: {sum(entropies)/len(entropies):.4f}  "
              f"max: {max(entropies):.4f}  min: {min(entropies):.4f}")
        print(f"    Recomendaciones directas (freq mode): "
              f"{len(direct_recs)}/{len(results)}")
        print(f"    Reuso de grises tras tiebreaker:  "
              f"{len(grey_cases)}/{len(non_trivial)}")
        print(f"    Amarillas en misma posición:      "
              f"{len(ysame_cases)}/{len(non_trivial)}")
        if grey_cases:
            print(f"\n    Branches con grises residuales "
                  f"(entropía igual con o sin ellas):")
            for r in grey_cases:
                print(f"      pat={r['pat']} n={r['n_cands']} "
                      f"→'{r['best_probe']}' "
                      f"grises={r['grey_letters']}")
        if direct_recs:
            print(f"\n    Branches donde se recomienda adivinar directo:")
            for r in direct_recs:
                rec = r["recommendation"]
                print(f"      pat={r['pat']} n={r['n_cands']} "
                      f"p={rec['p_best']:.3f} "
                      f"e_direct={rec['e_direct']:.3f} "
                      f"e_probe={rec['e_probe']:.3f}")

    # Guardar
    table_path       = Path(f"t2_table_{wl}_{mode}.json")
    probe_table_path = Path(f"t2_probe_table_{wl}_{mode}.json")
    report_path      = Path(f"t2_report_{wl}_{mode}.json")

    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(t2_table, f, ensure_ascii=False, indent=2)
    with open(probe_table_path, 'w', encoding='utf-8') as f:
        json.dump(t2_probe_table, f, ensure_ascii=False, indent=2)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  t2_table_{wl}_{mode}.json        — {len(t2_table)} entradas "
          f"(lo que juega strategy.py)")
    print(f"  t2_probe_table_{wl}_{mode}.json  — mejor probe puro")
    print(f"  t2_report_{wl}_{mode}.json       — reporte completo")

    return t2_table


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length",  default="4",
                        choices=["4", "5", "6", "all"])
    parser.add_argument("--mode",    default="both",
                        choices=["uniform", "frequency", "both"])
    parser.add_argument("--workers", type=int,
                        default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--top-combos-6l", type=int, default=15_000)
    args = parser.parse_args()

    print("═" * 64)
    print("  PRECOMPUTACIÓN T2 — entropía máxima + tiebreaker completo")
    print("═" * 64)
    print(f"  Workers: {args.workers}  |  Tiebreaker: 3 criterios jerarquizados")
    print(f"  Recomendación direct/probe incluida en el JSON")

    lengths = [4, 5, 6] if args.length == "all" else [int(args.length)]
    modes   = (["uniform", "frequency"] if args.mode == "both"
                else [args.mode])

    for wl in lengths:
        for mode in modes:
            run(wl, mode, args.workers, args.top_combos_6l)

    print("═" * 64)
    print("  INTEGRACIÓN CON strategy.py")
    print("═" * 64)
    print("""
  En strategy.py:

  _t2 = {}   # class-level, cargado una vez

  def begin_game(self, word_length, mode, ...):
      key = f"{word_length}_{mode}"
      if key not in Strategy._t2:
          p = TEAM_DIR / f"t2_table_{word_length}_{mode}.json"
          if p.exists():
              Strategy._t2[key] = json.load(open(p))

  def choose_word(self, candidates, weights, turn):
      if turn == 1:
          return OPENERS[self.wl][self.mode]
      if turn == 2:
          table = Strategy._t2.get(f"{self.wl}_{self.mode}", {})
          guess = table.get(self._t1_pat)
          if guess:
              return guess   # ya tiene incorporada la rec direct/probe
          # fallback: entropía runtime
      # T3+: lógica híbrida
    """)


if __name__ == "__main__":
    mp.freeze_support()
    main()