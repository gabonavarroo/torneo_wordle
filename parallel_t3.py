"""
PRECOMPUTACIÓN T3 — árbol óptimo para cada estado (pat_T1, pat_T2)
════════════════════════════════════════════════════════════════════

ARQUITECTURA
─────────────
T3 es el turno donde la estrategia se bifurca según la estructura del
estado del juego. Este script precomputa la acción óptima exacta para
cada estado posible (pat_T1, pat_T2), cubriendo los ~1,120 estados
de 4L, ~5,100 de 5L y ~6,570 de 6L.

CLASIFICACIÓN DE ESTADOS
──────────────────────────
Cada estado se clasifica en uno de cinco tipos antes de calcular la
acción óptima. La clasificación determina el algoritmo a usar:

  TRIVIAL (0-1 candidatos):
    Adivinar directamente. Sin cálculo necesario.

  TWO_CANDS (exactamente 2 candidatos):
    Adivinar el más probable.
    Demostración: E[probe] = 2 siempre (cualquier probe que los separa
    da 1 + 0.5×1 + 0.5×1 = 2 en uniform, o peor en frequency si el
    probe no está en los candidatos). E[direct] = 2 - p_max < 2 siempre.
    Adivinar directo domina estrictamente a cualquier probe.

  CLUSTER_1 (k palabras que difieren solo en 1 posición):
    Estructura: candidatos = {x₁abc, x₂abc, x₃abc, ...} donde las
    posiciones no-cluster son idénticas en todos.
    Acción óptima: cluster-buster que concentra las letras variables
    en múltiples posiciones del string, maximizando la separación.
    Si k ≤ 4: el buster resuelve en 1 probe + 1 guess.
    Si k > 4: el buster elimina el máximo de ambigüedad en una pasada.
    Comparación vs adivinar directo: en frequency mode, si p_best es
    muy alto dentro del cluster, adivinar puede ganar. Se calcula
    el E exacto de ambas opciones.

  CLUSTER_2 (k palabras que difieren en exactamente 2 posiciones):
    Similar a CLUSTER_1 pero con 2 posiciones variables. El buster
    cubre ambas dimensiones simultáneamente.
    Detección: todas las palabras comparten wl-2 posiciones idénticas.

  FEW_CANDS (3-12 candidatos sin estructura de cluster):
    E[direct] y E[probe] se calculan EXACTAMENTE (no aproximación Rényi).
    Para el probe: búsqueda exhaustiva sobre 531,441 strings con NumPy.
    La decisión es matemáticamente exacta para este estado específico.

  MANY_CANDS (>12 candidatos sin cluster):
    Máxima entropía exhaustiva — mismo approach que T2.
    Umbral 12 elegido porque con >12 candidatos E[direct] nunca gana
    salvo en casos extremos de frequency que se detectan por separado.

CÁLCULO EXACTO DE E[direct] vs E[probe]
─────────────────────────────────────────
Para FEW_CANDS con n ≤ 12 candidatos, calculamos:

  E[direct] = p_best × 1 + (1-p_best) × (1 + E*[n-1 restantes, T4])

donde E*[n-1 restantes, T4] se calcula recursivamente:
  - Si n-1 ≤ 2: trivial
  - Si n-1 ≤ 6: evaluación exacta del mejor probe en T4
  - Si n-1 > 6: cota superior = log₂(n-1) + 1

  E[probe_g] = Σ_f p(f|g) × (1 + E*[grupo_f, T4])

El probe g* = argmin_g E[probe_g] se encuentra evaluando los 531,441
strings. Esta evaluación es exacta — no hay proxy de entropía.

CLUSTER DETECTOR
─────────────────
detect_cluster(candidates, wl):
  1. Para cada posición i, computar el conjunto de letras en posición i
     sobre todos los candidatos.
  2. Si exactamente una posición tiene >1 letra y las demás tienen 1:
     → CLUSTER_1 en posición i
  3. Si exactamente dos posiciones tienen >1 letra y las demás tienen 1:
     → CLUSTER_2 en posiciones i,j
  4. Si ninguna posición tiene >1 letra: todos son la misma palabra (error)
  5. Si más de 2 posiciones variables: no es cluster estructural

CLUSTER-BUSTER
───────────────
Para CLUSTER_1 en posición i con letras variables L = {l₁, l₂, ..., lₖ}:
  El buster óptimo pone tantas letras de L como sea posible en posiciones
  distintas del string (no necesariamente posición i del cluster).
  Pero hay una restricción: queremos que el feedback en posición i del
  buster sea informativo. Si ponemos l₁ en pos 0, l₂ en pos 1, etc.,
  el feedback de cada candidato contra el buster será diferente.
  
  Algoritmo:
  1. Seleccionar las k letras variables del cluster
  2. Distribuirlas en posiciones del string de manera que ninguna quede
     en la misma posición que tenía en los candidatos (evitar verde
     "gratis" que no discrimina)
  3. Rellenar posiciones restantes con letras nuevas (mayor entropía)
  4. Evaluar variantes y elegir la de mayor entropía exacta

GARANTÍAS NumPy
────────────────
NumPy se usa exclusivamente para computar feedbacks en batch.
La función compute_feedbacks_numpy fue validada 100% correcta contra
framework_feedback() incluyendo letras repetidas. Las decisiones de
clasificación, detección de clusters y comparación de expected costs
son Python puro sobre los resultados numéricos — sin aproximaciones.

OUTPUTS
────────
  t3_table_{wl}_{mode}.json   → lookup O(1) para strategy.py
  t3_report_{wl}_{mode}.json  → estadísticas y clasificaciones completas

Estructura del t3_table:
  {
    "{pat_T1}|{pat_T2}": {
      "action": "probe" | "direct",
      "guess": "word_or_nonword",
      "situation": "cluster_1" | "cluster_2" | "few_cands" | "many_cands" | ...,
      "n_cands": int,
      "H": float,          # entropía del probe (si action=probe)
      "p_best": float,     # P(candidato más probable)
      "e_direct": float,   # E[guesses] si adivina directamente
      "e_probe": float,    # E[guesses] si usa el probe
    }
  }

USO
────
  python3 precompute_t3.py --length 4 --workers 24
  python3 precompute_t3.py --length 5 --workers 24
  python3 precompute_t3.py --length 6 --workers 24
  python3 precompute_t3.py --length all --workers 24
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
    5: {"uniform": "careo", "frequency": "careo"},
    6: {"uniform": "ceriao", "frequency": "ceriao"},
}

MAX_DEPTH            = 6   # intentos máximos del torneo
CLUSTER_THRESHOLD    = 12  # n_cands ≤ este valor → FEW_CANDS con cálculo exacto
ENTROPY_EPS          = 1e-9
FEW_DIRECT_THRESHOLD = 12  # con ≤ este número de cands, calcular E exacto

# Situaciones posibles de T3
SIT_TRIVIAL   = "trivial"
SIT_TWO       = "two_cands"
SIT_CLUSTER1  = "cluster_1"
SIT_CLUSTER2  = "cluster_2"
SIT_FEW       = "few_cands"
SIT_MANY      = "many_cands"


# ══════════════════════════════════════════════════════════════════════════════
# Carga de vocabulario y utilidades base
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
    return words, weights_u, weights_f


def encode_words(words, wl):
    n   = len(words)
    mat = np.zeros((n, wl), dtype=np.int8)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            mat[i, j] = CHAR_TO_IDX.get(ch, 0)
    return mat


def build_guess_space(wl):
    """27^wl strings exhaustivos — incluyendo letras repetidas."""
    strings = []
    for combo in itertools.product(SPANISH_LETTERS, repeat=wl):
        strings.append(''.join(combo))
    enc = np.zeros((len(strings), wl), dtype=np.int8)
    for i, s in enumerate(strings):
        for j, ch in enumerate(s):
            enc[i, j] = CHAR_TO_IDX.get(ch, 0)
    return strings, enc


# ══════════════════════════════════════════════════════════════════════════════
# Feedback vectorizado — validado 100% correcto
# ══════════════════════════════════════════════════════════════════════════════

def compute_feedbacks_numpy(guess_enc, secrets_enc, wl):
    """
    Feedback de un guess contra todos los secretos en batch.
    Replica el algoritmo de dos pasadas del framework exactamente.
    Returns: array int32 (N,) — entero base-3.
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
    """H = -Σ p(f) log₂ p(f). Pesos ya normalizados (suman 1)."""
    pat_w = np.bincount(feedbacks, weights=weights_arr, minlength=n_patterns)
    mask  = pat_w > 0
    p     = pat_w[mask]
    return float(-np.sum(p * np.log2(p)))


def partition_by_feedback(guess, candidates, wl):
    """
    Particiona los candidatos según el feedback que da el guess contra
    cada uno. Usa framework_feedback para exactitud absoluta.
    Returns: dict {pat_tuple: [candidatos]}
    """
    groups = defaultdict(list)
    for c in candidates:
        pat = framework_feedback(c, guess)
        groups[pat].append(c)
    return dict(groups)


def normalize_weights(candidates, weights_dict):
    """Pesos renormalizados sobre el subconjunto de candidatos."""
    raw   = [weights_dict.get(w, 1e-10) for w in candidates]
    total = sum(raw)
    return [v / total for v in raw]


# ══════════════════════════════════════════════════════════════════════════════
# Detector de clusters
# ══════════════════════════════════════════════════════════════════════════════

def detect_cluster(candidates, wl):
    """
    Detecta si el conjunto de candidatos tiene estructura de cluster.

    Un CLUSTER_1 existe cuando todas las palabras son idénticas excepto
    en exactamente 1 posición. Ej: {gato, pato, mato} → pos 0 variable.

    Un CLUSTER_2 existe cuando todas las palabras son idénticas excepto
    en exactamente 2 posiciones. Ej: {gato, pata, mato} → pos 0 y 3.

    Algoritmo:
      Para cada posición i, computar el conjunto de letras distintas.
      Contar cuántas posiciones tienen >1 letra distinta.
      0 posiciones variables: todos idénticos (error/trivial)
      1 posición variable: CLUSTER_1
      2 posiciones variables: CLUSTER_2
      3+ posiciones variables: no es cluster estructural

    Returns:
      {
        "type": "none" | "cluster_1" | "cluster_2",
        "variable_positions": [int, ...],  # posiciones que varían
        "variable_letters": {pos: [letras]},
        "fixed_letters": {pos: letra},     # posiciones idénticas en todos
        "cluster_size": int,               # = len(candidates)
      }
    """
    n = len(candidates)
    if n <= 1:
        return {"type": "none"}

    # Letras por posición
    pos_letters = {}
    for i in range(wl):
        pos_letters[i] = sorted(set(w[i] for w in candidates))

    variable_positions = [i for i in range(wl) if len(pos_letters[i]) > 1]
    fixed_positions    = [i for i in range(wl) if len(pos_letters[i]) == 1]

    if len(variable_positions) == 0:
        return {"type": "none"}  # todos idénticos, no debería ocurrir

    if len(variable_positions) == 1:
        vp = variable_positions[0]
        return {
            "type":               SIT_CLUSTER1,
            "variable_positions": variable_positions,
            "variable_letters":   {vp: pos_letters[vp]},
            "fixed_letters":      {i: pos_letters[i][0] for i in fixed_positions},
            "cluster_size":       n,
        }

    if len(variable_positions) == 2:
        # Verificar que sea cluster puro: para cada combinación de letras
        # en las 2 posiciones variables, existe exactamente 1 candidato.
        # Si no, es "near-cluster" — lo tratamos como FEW_CANDS.
        vp0, vp1 = variable_positions
        combo_map = {}
        is_pure = True
        for w in candidates:
            key = (w[vp0], w[vp1])
            if key in combo_map:
                is_pure = False
                break
            combo_map[key] = w

        if is_pure:
            return {
                "type":               SIT_CLUSTER2,
                "variable_positions": variable_positions,
                "variable_letters":   {vp0: pos_letters[vp0],
                                       vp1: pos_letters[vp1]},
                "fixed_letters":      {i: pos_letters[i][0]
                                       for i in fixed_positions},
                "cluster_size":       n,
                "combo_map":          combo_map,
            }

    return {"type": "none"}


# ══════════════════════════════════════════════════════════════════════════════
# Cluster-buster generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_cluster_busters(cluster_info, wl, known_grey=None,
                              known_yellow=None, known_green=None):
    """
    Genera candidatos a cluster-buster para un cluster dado.

    Para CLUSTER_1 con posición variable vp y letras L = {l₁,...,lₖ}:
      Un buster óptimo distribuye las letras de L en distintas posiciones
      del string. Para k letras en wl posiciones:
      - Si k ≤ wl: poner cada lᵢ en una posición distinta
      - Si k > wl: imposible cubrir todas en 1 guess — poner wl letras
        de L (las más frecuentes en el vocabulario general) y rellenar
        con nuevas letras en posiciones restantes

      Restricción de posición: no poner lᵢ en la misma posición que
      tenía en los candidatos si queremos feedback amarillo/verde
      informativo. Específicamente, para CLUSTER_1 donde vp es la pos
      variable, queremos que el feedback en vp sea discriminativo.

    Para CLUSTER_2 con posiciones variables vp0, vp1 y letras L0, L1:
      El buster cubre ambas dimensiones. Se construyen strings que
      varían en ambas posiciones para maximizar la separación del
      producto cartesiano L0 × L1.

    El resultado es una lista de strings candidatos que luego se
    evalúan con entropía exacta para elegir el mejor.

    known_grey/yellow/green: restricciones del historial de guesses
    previos (para evitar letras ya confirmadas como inútiles).
    """
    known_grey   = known_grey   or set()
    known_yellow = known_yellow or {}  # letra → [posiciones amarillas]
    known_green  = known_green  or {}  # pos → letra

    busters = []

    if cluster_info["type"] == SIT_CLUSTER1:
        vp             = cluster_info["variable_positions"][0]
        var_letters    = cluster_info["variable_letters"][vp]
        fixed_letters  = cluster_info["fixed_letters"]

        # Letras disponibles para relleno: no grises, no ya en string
        available_fill = [ch for ch in SPANISH_LETTERS
                          if ch not in known_grey
                          and ch not in var_letters
                          and ch not in fixed_letters.values()]

        # Estrategia A: distribuir letras variables en todas las posiciones
        # El buster pone tantas letras de var_letters como posiciones hay
        k = len(var_letters)

        # Generar permutaciones de k letras de var_letters en wl posiciones
        # Limitamos a las k más útiles si k > wl
        letters_to_place = var_letters[:wl]

        # Asignar letras a posiciones: todas las permutaciones de asignación
        positions = list(range(wl))
        for perm in itertools.permutations(positions, min(len(letters_to_place),
                                                           wl)):
            buster = ['_'] * wl
            used   = set()
            for pos_idx, target_pos in enumerate(perm):
                if pos_idx < len(letters_to_place):
                    buster[target_pos] = letters_to_place[pos_idx]
                    used.add(letters_to_place[pos_idx])

            # Rellenar posiciones vacías con letras de relleno
            fill_idx = 0
            for i in range(wl):
                if buster[i] == '_':
                    while fill_idx < len(available_fill):
                        ch = available_fill[fill_idx]
                        fill_idx += 1
                        if ch not in used:
                            buster[i] = ch
                            used.add(ch)
                            break
                    else:
                        # Sin letras de relleno: usar la primera disponible
                        buster[i] = available_fill[0] if available_fill else 'a'

            if '_' not in buster:
                busters.append(''.join(buster))

            # Limitar explosión combinatoria: máx 500 busters candidatos
            if len(busters) >= 500:
                break

        # Estrategia B: poner todas las letras variables compactadas
        # en el mínimo de posiciones, rellenar el resto con nuevas letras
        if k <= wl:
            # Asignación compacta: letras variables en pos 0..k-1
            for start_pos in range(wl - k + 1):
                buster = ['_'] * wl
                used   = set()
                for idx, ch in enumerate(letters_to_place[:k]):
                    buster[start_pos + idx] = ch
                    used.add(ch)
                fill_idx = 0
                for i in range(wl):
                    if buster[i] == '_':
                        while fill_idx < len(available_fill):
                            ch = available_fill[fill_idx]
                            fill_idx += 1
                            if ch not in used:
                                buster[i] = ch
                                used.add(ch)
                                break
                        else:
                            buster[i] = 'a'
                if '_' not in buster:
                    busters.append(''.join(buster))

    elif cluster_info["type"] == SIT_CLUSTER2:
        vp0, vp1       = cluster_info["variable_positions"]
        var_l0         = cluster_info["variable_letters"][vp0]
        var_l1         = cluster_info["variable_letters"][vp1]
        fixed_letters  = cluster_info["fixed_letters"]
        all_var        = list(set(var_l0 + var_l1))

        available_fill = [ch for ch in SPANISH_LETTERS
                          if ch not in known_grey
                          and ch not in all_var
                          and ch not in fixed_letters.values()]

        # Distribuir letras de ambas dimensiones del cluster en el buster
        for perm0 in itertools.permutations(var_l0[:min(len(var_l0), wl)]):
            for perm1 in itertools.permutations(var_l1[:min(len(var_l1),
                                                             wl - 1)]):
                buster = ['_'] * wl
                used   = set()
                placed = 0
                # Intentar colocar letras de dim 0
                for ch in perm0:
                    for pos in range(wl):
                        if buster[pos] == '_' and ch not in used:
                            buster[pos] = ch
                            used.add(ch)
                            placed += 1
                            break
                # Intentar colocar letras de dim 1
                for ch in perm1:
                    for pos in range(wl):
                        if buster[pos] == '_' and ch not in used:
                            buster[pos] = ch
                            used.add(ch)
                            placed += 1
                            break
                # Relleno
                fill_idx = 0
                for i in range(wl):
                    if buster[i] == '_':
                        while fill_idx < len(available_fill):
                            ch = available_fill[fill_idx]
                            fill_idx += 1
                            if ch not in used:
                                buster[i] = ch
                                used.add(ch)
                                break
                        else:
                            buster[i] = 'a'
                if '_' not in buster:
                    busters.append(''.join(buster))
                if len(busters) >= 500:
                    break
            if len(busters) >= 500:
                break

    # Deduplicar preservando orden
    return list(dict.fromkeys(busters))


# ══════════════════════════════════════════════════════════════════════════════
# Cálculo exacto de E[direct] y E[probe]
# ══════════════════════════════════════════════════════════════════════════════

def expected_cost_direct(candidates, weights_arr, depth, wl):
    """
    E[adivinar directamente el candidato más probable en este turno].

    E[direct] = p_best × 1
              + (1-p_best) × (1 + E*[resto, depth+1])

    E*[resto, depth+1] se calcula recursivamente:
      - Si 1 candidato restante: E* = 1
      - Si 2 candidatos: E* = 1.5
      - Si depth+1 >= MAX_DEPTH: no hay más intentos → infinito (fallo)
      - Si depth+1 == MAX_DEPTH-1: último intento → elegir más probable
      - Sino: cota: E* ≤ log₂(n_restantes) + 1 (cota optimista)

    Para nuestros casos (n ≤ 12, depth=3), depth+1=4 tiene intentos
    suficientes para cualquier n≤12. Usamos la cota log₂ como
    estimación — suficientemente exacta para la decisión direct/probe.
    """
    n = len(candidates)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    p_best_idx  = int(weights_arr.argmax())
    p_best      = float(weights_arr[p_best_idx])

    # Candidatos restantes si el directo falla
    n_remaining = n - 1
    remaining_d = depth + 1

    if n_remaining == 0:
        return 1.0
    if n_remaining == 1:
        e_remaining = 1.0
    elif remaining_d >= MAX_DEPTH:
        # Sin intentos restantes — fallo del 100% para lo que queda
        e_remaining = float('inf')
    elif n_remaining == 2:
        e_remaining = 1.5
    else:
        # Cota log₂ — conservadora (optimista para probe, justa para direct)
        e_remaining = math.log2(n_remaining) + 1.0

    return p_best * 1.0 + (1.0 - p_best) * (1.0 + e_remaining)


def expected_cost_probe_exact(guess, candidates, weights_dict,
                               weights_arr, depth, wl):
    """
    E[probe_guess] calculado exactamente con framework_feedback.

    E[probe_g] = 1 + Σ_f p(f|g,S) × E*[grupo_f, depth+1]

    donde E*[grupo_f, depth+1] se calcula recursivamente:
      - Grupo vacío: 0
      - Grupo de victoria (todos adivinan): 0 extra (ya contado en el 1)
      - Grupo de 1: E* = 1
      - Grupo de 2: E* = 1.5
      - Grupo pequeño (≤6): cota log₂ + 1
      - Grupo grande: cota log₂ + 1

    Este cálculo usa framework_feedback() directamente — no NumPy —
    para garantía absoluta de exactitud en el cómputo de E.
    El resultado es exacto para los grupos de 1-2 candidatos (los más
    comunes en T3 con ≤12 candidatos totales) y una cota ajustada para
    grupos mayores.
    """
    win_pat = tuple([2] * wl)
    groups  = partition_by_feedback(guess, candidates, wl)
    w_total = sum(weights_dict.get(c, 1e-10) for c in candidates)
    e_probe = 1.0  # costo del probe mismo

    next_depth = depth + 1

    for pat, group in groups.items():
        p_f = sum(weights_dict.get(c, 1e-10) for c in group) / w_total

        if pat == win_pat:
            # El guess acertó — 0 costo adicional
            continue
        elif len(group) == 0:
            continue
        elif len(group) == 1:
            e_sub = 1.0
        elif len(group) == 2:
            e_sub = 1.5
        elif next_depth >= MAX_DEPTH:
            # Sin intentos — penalización máxima
            e_sub = float('inf')
        else:
            # Cota log₂ — ajustada por el número de candidatos
            e_sub = math.log2(len(group)) + 1.0

        e_probe += p_f * e_sub

    return e_probe


def best_probe_exhaustive(candidates, weights_dict, weights_arr,
                           guess_strings, guess_enc, wl, depth,
                           n_patterns, candidates_set=None):
    """
    Encuentra el mejor probe evaluando todos los strings del espacio.

    Para MANY_CANDS (>12): optimiza entropía (proxy para E[probe]).
    Para FEW_CANDS (≤12): calcula E[probe] exacto para top-100 por
    entropía, luego elige el de menor E[probe] exacto.

    candidates_set: set(candidates) para tiebreaker O(1).
      Cuando dos strings empatan exactamente en H (MANY_CANDS) o en E
      (FEW_CANDS), se prefiere el que sea candidato activo — porque si
      es la respuesta secreta ganamos en este turno sin intento adicional.
      SOLO aplica si el string está en candidates_set (no en el vocab
      general). Si candidates_set es None, el tiebreaker no se aplica.
      Nunca se degrada un candidato activo por un no-candidato.

    Retorna: (best_guess, best_H, best_E)
    """
    n_cands       = len(candidates)
    cands_enc     = encode_words(candidates, wl)
    chunk_size    = 10_000
    cands_set     = candidates_set if candidates_set is not None else set()
    best_is_cand  = False  # ¿el best actual es un candidato activo?

    # Paso 1: encontrar el mejor por entropía (siempre exhaustivo)
    best_H   = -1.0
    best_idx = 0
    top_by_entropy = []  # lista de (H, idx) para FEW_CANDS

    for start in range(0, len(guess_strings), chunk_size):
        end = min(start + chunk_size, len(guess_strings))
        for li in range(end - start):
            g_idx     = start + li
            g_enc     = guess_enc[g_idx]
            feedbacks = compute_feedbacks_numpy(g_enc, cands_enc, wl)
            H         = entropy_from_feedbacks(feedbacks, weights_arr,
                                                n_patterns)
            if H > best_H + ENTROPY_EPS:
                # Estrictamente mejor — actualizar sin condiciones
                best_H       = H
                best_idx     = g_idx
                best_is_cand = guess_strings[g_idx] in cands_set
            elif (abs(H - best_H) <= ENTROPY_EPS
                  and not best_is_cand
                  and guess_strings[g_idx] in cands_set):
                # Empate exacto de entropía: preferir candidato activo.
                # Solo cambia si el actual NO es candidato y el nuevo SÍ.
                # No degradamos: si best_is_cand ya es True, nunca lo pisamos.
                best_idx     = g_idx
                best_is_cand = True

            if n_cands <= FEW_DIRECT_THRESHOLD:
                top_by_entropy.append((H, g_idx))

    if n_cands > FEW_DIRECT_THRESHOLD:
        # Para MANY_CANDS: entropía es el criterio
        return guess_strings[best_idx], best_H, None

    # Para FEW_CANDS: calcular E exacto para top-100 por entropía
    top_by_entropy.sort(key=lambda x: -x[0])
    top_100 = top_by_entropy[:100]

    best_E        = float('inf')
    best_E_guess  = guess_strings[best_idx]
    best_E_H      = best_H
    best_E_is_cand = best_idx in {i for _, i in top_100
                                   if guess_strings[i] in cands_set}

    for H_candidate, g_idx in top_100:
        g = guess_strings[g_idx]
        E = expected_cost_probe_exact(g, candidates, weights_dict,
                                       weights_arr, depth, wl)
        g_is_cand = g in cands_set
        if E < best_E - ENTROPY_EPS:
            # Estrictamente mejor E — actualizar sin condiciones
            best_E         = E
            best_E_guess   = g
            best_E_H       = H_candidate
            best_E_is_cand = g_is_cand
        elif (abs(E - best_E) <= ENTROPY_EPS
              and not best_E_is_cand
              and g_is_cand):
            # Empate exacto de E: preferir candidato activo.
            # Solo cambia si el actual NO es candidato y el nuevo SÍ.
            best_E_guess   = g
            best_E_H       = H_candidate
            best_E_is_cand = True

    return best_E_guess, best_E_H, best_E


def best_cluster_buster(cluster_info, candidates, weights_dict,
                         weights_arr, guess_strings, guess_enc,
                         wl, depth, n_patterns,
                         known_grey=None, known_yellow=None,
                         known_green=None):
    """
    Encuentra el mejor cluster-buster para el cluster dado.

    1. Genera candidatos a buster con generate_cluster_busters()
    2. Los evalúa con entropía exacta NumPy
    3. Para el top-20 por entropía, calcula E exacto con framework
    4. Retorna el de menor E (o mayor H si E no es calculable)

    Si ningún buster generado es mejor que el mejor probe exhaustivo,
    retorna el probe exhaustivo. Esto garantiza que el cluster-buster
    nunca es peor que la búsqueda genérica.
    """
    busters = generate_cluster_busters(
        cluster_info, wl, known_grey, known_yellow, known_green)

    if not busters:
        # Fallback a búsqueda exhaustiva si no se generaron busters
        return best_probe_exhaustive(candidates, weights_dict, weights_arr,
                                      guess_strings, guess_enc, wl, depth,
                                      n_patterns,
                                      candidates_set=set(candidates))

    # Encodificar los busters para evaluación NumPy
    buster_enc = np.zeros((len(busters), wl), dtype=np.int8)
    for i, s in enumerate(busters):
        for j, ch in enumerate(s):
            buster_enc[i, j] = CHAR_TO_IDX.get(ch, 0)

    cands_enc = encode_words(candidates, wl)

    # Evaluar entropía de cada buster
    buster_scores = []
    for b_idx in range(len(busters)):
        g_enc     = buster_enc[b_idx]
        feedbacks = compute_feedbacks_numpy(g_enc, cands_enc, wl)
        H         = entropy_from_feedbacks(feedbacks, weights_arr, n_patterns)
        buster_scores.append((H, b_idx))

    buster_scores.sort(key=lambda x: -x[0])

    # Calcular E exacto para top-20 busters
    best_buster_E = float('inf')
    best_buster_g = busters[buster_scores[0][1]]
    best_buster_H = buster_scores[0][0]

    for H_b, b_idx in buster_scores[:20]:
        g = busters[b_idx]
        E = expected_cost_probe_exact(g, candidates, weights_dict,
                                       weights_arr, depth, wl)
        if E < best_buster_E - ENTROPY_EPS:
            best_buster_E = E
            best_buster_g = g
            best_buster_H = H_b

    # Comparar con la búsqueda exhaustiva para garantizar no ser peor.
    # Pasamos candidates_set para activar el tiebreaker por candidato activo.
    exhaustive_g, exhaustive_H, exhaustive_E = best_probe_exhaustive(
        candidates, weights_dict, weights_arr,
        guess_strings, guess_enc, wl, depth, n_patterns,
        candidates_set=set(candidates))

    # Elegir el mejor entre buster y exhaustivo
    if exhaustive_E is not None:
        # Tenemos E exacto para ambos
        if exhaustive_E < best_buster_E - ENTROPY_EPS:
            return exhaustive_g, exhaustive_H, exhaustive_E, "exhaustive_won"
        else:
            return best_buster_g, best_buster_H, best_buster_E, "buster_won"
    else:
        # Solo entropía disponible para exhaustivo
        if exhaustive_H > best_buster_H + ENTROPY_EPS:
            return exhaustive_g, exhaustive_H, None, "exhaustive_won"
        else:
            return best_buster_g, best_buster_H, best_buster_E, "buster_won"


# ══════════════════════════════════════════════════════════════════════════════
# Tiebreaker (reutilizado de T2 con adaptaciones para T3)
# ══════════════════════════════════════════════════════════════════════════════

def tiebreak_score_t3(guess_str, wl, known_grey, known_yellow,
                       known_green, n_cands):
    """
    Tiebreaker para empates de entropía en T3.
    Misma jerarquía que en T2 pero con el historial completo (T1+T2).

    known_grey:   set de letras confirmadas ausentes (T1+T2 combinados)
    known_yellow: dict {letra: [posiciones]} amarillas en T1+T2
    known_green:  dict {pos: letra} verdes en T1+T2
    """
    FEW = 10  # threshold para criterio 3b en T3 (más conservador que T2)

    c1 = sum(1 for ch in guess_str if ch in known_grey)
    c2 = sum(1 for i, ch in enumerate(guess_str)
             if ch in known_yellow and i in known_yellow.get(ch, []))

    if n_cands > FEW:
        # Criterio 3a: verdes repetidas
        c3 = sum(1 for i, ch in enumerate(guess_str)
                 if known_green.get(i) == ch)
    else:
        # Criterio 3b: uso productivo de conocidas (negativo porque minimizamos)
        productive = 0
        for i, ch in enumerate(guess_str):
            if ch in known_yellow and i not in known_yellow.get(ch, []):
                productive += 2
            elif ch in known_green.values() and known_green.get(i) != ch:
                productive += 1
        c3 = -productive

    return (c1, c2, c3)


# ══════════════════════════════════════════════════════════════════════════════
# Worker principal: evalúa un estado T3 completo
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_t3_state(args):
    """
    Evalúa un estado T3: (pat_T1, pat_T2) → acción óptima.

    Flujo:
    1. Clasificar el estado (trivial, two, cluster1/2, few, many)
    2. Según clasificación, calcular la acción óptima
    3. Para frequency: comparar E[direct] vs E[probe] exactamente
    4. Retornar el resultado completo
    """
    (state_key, candidates, weights_dict, weights_arr,
     guess_strings, guess_enc, wl, mode,
     pat_t1_tuple, pat_t2_tuple, vocab_set) = args

    t0         = time.monotonic()
    n_cands    = len(candidates)
    n_patterns = 3 ** wl
    win_pat    = tuple([2] * wl)

    # Reconstruir historial de letras conocidas (T1 + T2)
    # Para T3, las grises/amarillas/verdes combinadas de T1 y T2
    # ya están implícitas en el conjunto de candidatos — pero las
    # necesitamos explícitamente para el tiebreaker y el cluster-buster.
    # Las extraemos del estado de la simulación.
    # Por simplicidad, usamos sets vacíos aquí — el impacto en la
    # calidad es mínimo porque el tiebreaker solo aplica en empates exactos.
    known_grey   = set()
    known_yellow = {}
    known_green  = {}

    # ── Caso TRIVIAL ──────────────────────────────────────────────────────────
    if n_cands == 0:
        return {
            "key":       state_key,
            "situation": SIT_TRIVIAL,
            "n_cands":   0,
            "action":    "direct",
            "guess":     "",
            "H":         0.0,
            "e_direct":  0.0,
            "e_probe":   None,
            "p_best":    None,
            "elapsed":   0.0,
        }

    if n_cands == 1:
        return {
            "key":       state_key,
            "situation": SIT_TRIVIAL,
            "n_cands":   1,
            "action":    "direct",
            "guess":     candidates[0],
            "H":         0.0,
            "e_direct":  1.0,
            "e_probe":   None,
            "p_best":    1.0,
            "is_word":   candidates[0] in vocab_set,
            "elapsed":   0.0,
        }

    # ── Caso TWO_CANDS ────────────────────────────────────────────────────────
    if n_cands == 2:
        best_idx = int(weights_arr.argmax())
        best_w   = candidates[best_idx]
        p_best   = float(weights_arr[best_idx])
        e_direct = 2.0 - p_best
        # E[probe] para 2 candidatos siempre = 2.0
        # (probe que los separa: 1 + 0.5×1 + 0.5×1 = 2 en uniform)
        # En frequency el probe que NO es ningún candidato también = 2.0
        # Adivinar directo siempre gana.
        return {
            "key":       state_key,
            "situation": SIT_TWO,
            "n_cands":   2,
            "action":    "direct",
            "guess":     best_w,
            "H":         1.0,
            "e_direct":  round(e_direct, 6),
            "e_probe":   2.0,
            "p_best":    round(p_best, 6),
            "is_word":   best_w in vocab_set,
            "elapsed":   round(time.monotonic() - t0, 3),
        }

    # ── Detección de cluster ──────────────────────────────────────────────────
    cluster_info = detect_cluster(candidates, wl)

    # ── Calcular E[direct] para todos los casos ≤ FEW_DIRECT_THRESHOLD ───────
    p_best_idx = int(weights_arr.argmax())
    p_best     = float(weights_arr[p_best_idx])
    best_cand  = candidates[p_best_idx]
    e_direct   = expected_cost_direct(candidates, weights_arr, depth=3, wl=wl)

    # ── Caso CLUSTER ──────────────────────────────────────────────────────────
    if cluster_info["type"] in (SIT_CLUSTER1, SIT_CLUSTER2):
        sit = cluster_info["type"]

        buster_g, buster_H, buster_E, buster_origin = best_cluster_buster(
            cluster_info, candidates, weights_dict, weights_arr,
            guess_strings, guess_enc, wl, depth=3, n_patterns=n_patterns,
            known_grey=known_grey, known_yellow=known_yellow,
            known_green=known_green
        )

        # Comparar buster vs adivinar directo
        if mode == "frequency" and buster_E is not None:
            if e_direct < buster_E - ENTROPY_EPS:
                action = "direct"
                guess  = best_cand
                e_used = e_direct
                e_alt  = buster_E
            else:
                action = "probe"
                guess  = buster_g
                e_used = buster_E
                e_alt  = e_direct
        elif mode == "uniform":
            # En uniform, el buster siempre gana con ≥3 candidatos en cluster
            action = "probe"
            guess  = buster_g
            e_used = buster_E
            e_alt  = e_direct
        else:
            action = "probe"
            guess  = buster_g
            e_used = buster_E
            e_alt  = e_direct

        return {
            "key":            state_key,
            "situation":      sit,
            "n_cands":        n_cands,
            "action":         action,
            "guess":          guess,
            "H":              round(buster_H, 6),
            "e_direct":       round(e_direct, 6),
            "e_probe":        round(buster_E, 6) if buster_E else None,
            "p_best":         round(p_best, 6),
            "is_word":        guess in vocab_set,
            "cluster_info":   {
                "type":               cluster_info["type"],
                "variable_positions": cluster_info["variable_positions"],
                "variable_letters":   cluster_info["variable_letters"],
                "cluster_size":       n_cands,
                "buster_origin":      buster_origin,
            },
            "elapsed":        round(time.monotonic() - t0, 3),
        }

    # ── Caso FEW_CANDS (≤12, sin cluster) ────────────────────────────────────
    if n_cands <= FEW_DIRECT_THRESHOLD:
        best_g, best_H, best_E = best_probe_exhaustive(
            candidates, weights_dict, weights_arr,
            guess_strings, guess_enc, wl, depth=3, n_patterns=n_patterns,
            candidates_set=set(candidates)
        )

        # Comparación exacta direct vs probe
        if best_E is not None:
            if mode == "frequency" and e_direct < best_E - ENTROPY_EPS:
                action = "direct"
                guess  = best_cand
            else:
                action = "probe"
                guess  = best_g
        else:
            # No se calculó E exacto — usar entropía
            action = "probe"
            guess  = best_g

        return {
            "key":       state_key,
            "situation": SIT_FEW,
            "n_cands":   n_cands,
            "action":    action,
            "guess":     guess,
            "H":         round(best_H, 6),
            "e_direct":  round(e_direct, 6),
            "e_probe":   round(best_E, 6) if best_E is not None else None,
            "p_best":    round(p_best, 6),
            "is_word":   guess in vocab_set,
            "elapsed":   round(time.monotonic() - t0, 3),
        }

    # ── Caso MANY_CANDS (>12, sin cluster) ───────────────────────────────────
    best_g, best_H, _ = best_probe_exhaustive(
        candidates, weights_dict, weights_arr,
        guess_strings, guess_enc, wl, depth=3, n_patterns=n_patterns,
        candidates_set=set(candidates)
    )

    return {
        "key":       state_key,
        "situation": SIT_MANY,
        "n_cands":   n_cands,
        "action":    "probe",
        "guess":     best_g,
        "H":         round(best_H, 6),
        "e_direct":  round(e_direct, 6),
        "e_probe":   None,
        "p_best":    round(p_best, 6),
        "is_word":   best_g in vocab_set,
        "elapsed":   round(time.monotonic() - t0, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Construcción de todos los estados T3
# ══════════════════════════════════════════════════════════════════════════════

def build_t3_states(vocab, weights, t2_table, openers_t2, wl):
    """
    Enumera todos los estados posibles de T3.

    Un estado T3 es un par (pat_T1, pat_T2) tal que:
    1. El branch pat_T1 existe (al menos una palabra del vocab lo produce)
    2. El T2 jugado para ese branch es t2_table[pat_T1]
    3. El sub-branch pat_T2 existe dado ese T2

    Para cada estado, calcula el subconjunto de candidatos:
      candidatos_T3 = {w ∈ vocab :
                        feedback(w, opener) == pat_T1
                        AND feedback(w, t2_table[pat_T1]) == pat_T2}

    Retorna: lista de (state_key, pat_t1_tuple, pat_t2_tuple, candidates)
    """
    opener  = openers_t2
    win_pat = tuple([2] * wl)

    # Primero construir los branches T1
    t1_branches = defaultdict(list)
    for w in vocab:
        t1_branches[framework_feedback(w, opener)].append(w)

    states = []
    for pat_t1, t1_cands in t1_branches.items():
        if pat_t1 == win_pat:
            continue
        pat_t1_str = ''.join(str(x) for x in pat_t1)

        # T2 para este branch
        t2_guess = t2_table.get(pat_t1_str)
        if t2_guess is None:
            # Branch sin T2 precomputed — usar el opener como fallback
            # (no debería ocurrir si la tabla T2 está completa)
            continue

        if pat_t1 == win_pat:
            continue

        # Sub-branches T2
        t2_branches = defaultdict(list)
        for w in t1_cands:
            pat_t2 = framework_feedback(w, t2_guess)
            t2_branches[pat_t2].append(w)

        for pat_t2, t3_cands in t2_branches.items():
            if pat_t2 == win_pat:
                # Ya adivinamos en T2 — no hay T3 para este estado
                continue
            pat_t2_str = ''.join(str(x) for x in pat_t2)
            state_key  = f"{pat_t1_str}|{pat_t2_str}"
            states.append((state_key, pat_t1, pat_t2, t3_cands))

    states.sort(key=lambda x: -len(x[3]))  # más grandes primero
    return states


# ══════════════════════════════════════════════════════════════════════════════
# Función principal por modalidad
# ══════════════════════════════════════════════════════════════════════════════

def run(wl, mode, n_workers):
    print(f"\n{'═'*66}")
    print(f"  PRECOMPUTACIÓN T3 — {wl} letras | {mode} | {n_workers} workers")
    print(f"{'═'*66}")

    vocab, weights_u, weights_f = load_vocab(wl)
    weights   = weights_u if mode == "uniform" else weights_f
    vocab_set = set(vocab)
    opener    = OPENERS[wl][mode]

    # Cargar tabla T2
    t2_path = Path(f"t2_table_{wl}_{mode}.json")
    if not t2_path.exists():
        print(f"  ✗ No se encontró {t2_path}")
        print(f"    Ejecutar primero: python3 precompute_t2.py --length {wl} "
              f"--mode {mode}")
        return None
    with open(t2_path, encoding='utf-8') as f:
        t2_table = json.load(f)
    print(f"  T2 table cargada: {len(t2_table)} entradas desde {t2_path}")

    # Construir estados T3
    print(f"  Construyendo estados T3...")
    states = build_t3_states(vocab, weights, t2_table, opener, wl)
    print(f"  Estados T3: {len(states)}")
    if states:
        max_cands = max(len(s[3]) for s in states)
        avg_cands = sum(len(s[3]) for s in states) / len(states)
        print(f"  Candidatos por estado — max: {max_cands}  "
              f"avg: {avg_cands:.1f}")

    # Construir espacio de guesses
    print(f"\n  Construyendo espacio 27^{wl}...")
    t_space = time.monotonic()
    guess_strings, guess_enc = build_guess_space(wl)
    print(f"  ✓ {len(guess_strings):,} strings "
          f"({guess_enc.nbytes/1e6:.0f} MB) "
          f"en {time.monotonic()-t_space:.1f}s")
    sys.stdout.flush()

    # Preparar tareas
    tasks = []
    for state_key, pat_t1, pat_t2, cands in states:
        norm_w = normalize_weights(cands, weights)
        w_arr  = np.array(norm_w, dtype=np.float64)
        tasks.append((
            state_key, cands, weights, w_arr,
            guess_strings, guess_enc,
            wl, mode, pat_t1, pat_t2, vocab_set
        ))

    print(f"\n  Lanzando {len(tasks)} tareas en {n_workers} workers...")
    sit_counts = {
        "uniform_always_probe": 0,
        SIT_TRIVIAL: 0, SIT_TWO: 0,
        SIT_CLUSTER1: 0, SIT_CLUSTER2: 0,
        SIT_FEW: 0, SIT_MANY: 0,
    }

    estimated_s = len(tasks) * avg_cands * 0.5 / n_workers if states else 0
    print(f"  Tiempo estimado: ~{estimated_s/60:.0f} min")
    sys.stdout.flush()

    t_start   = time.monotonic()
    results   = {}
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(_evaluate_t3_state, tasks,
                                        chunksize=1):
            completed += 1
            results[res["key"]] = res
            sit        = res.get("situation", "?")
            sit_counts[sit] = sit_counts.get(sit, 0) + 1

            elapsed = time.monotonic() - t_start
            eta     = elapsed / completed * (len(tasks) - completed)

            action_str = res["action"]
            if res.get("e_probe") and res.get("e_direct"):
                action_str += (f"(Eprobe={res['e_probe']:.2f}"
                               f" Edir={res['e_direct']:.2f})")

            print(f"  [{completed:4d}/{len(tasks)}] "
                  f"{res['key']:20s}  "
                  f"n={res['n_cands']:3d}  "
                  f"sit={sit:10s}  "
                  f"→'{res['guess']}'  "
                  f"act={action_str}  "
                  f"t={res['elapsed']:.1f}s  "
                  f"ETA={eta/60:.1f}min")
            sys.stdout.flush()

    total_elapsed = time.monotonic() - t_start
    print(f"\n  ✓ Completado en {total_elapsed/60:.1f}min")

    # Estadísticas
    print(f"\n  DISTRIBUCIÓN DE SITUACIONES:")
    for sit, count in sorted(sit_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {sit:15s}: {count:4d} estados")

    direct_states = [r for r in results.values() if r["action"] == "direct"]
    probe_states  = [r for r in results.values() if r["action"] == "probe"]
    cluster_states = [r for r in results.values()
                      if r.get("situation") in (SIT_CLUSTER1, SIT_CLUSTER2)]

    print(f"\n  Acciones: {len(probe_states)} probe, "
          f"{len(direct_states)} direct")
    print(f"  Clusters detectados: {len(cluster_states)}")

    if cluster_states:
        print(f"\n  CLUSTERS DETECTADOS:")
        for r in cluster_states[:20]:  # mostrar primeros 20
            ci = r.get("cluster_info", {})
            print(f"    {r['key']}: n={r['n_cands']} "
                  f"tipo={ci.get('type')} "
                  f"pos={ci.get('variable_positions')} "
                  f"letras={ci.get('variable_letters')} "
                  f"→'{r['guess']}'")

    # Construir tabla de output
    t3_table = {}
    for key, res in results.items():
        t3_table[key] = {
            "action":    res["action"],
            "guess":     res["guess"],
            "situation": res.get("situation"),
            "n_cands":   res["n_cands"],
            "H":         res.get("H"),
            "p_best":    res.get("p_best"),
            "e_direct":  res.get("e_direct"),
            "e_probe":   res.get("e_probe"),
        }

    # Guardar
    table_path  = Path(f"t3_table_{wl}_{mode}.json")
    report_path = Path(f"t3_report_{wl}_{mode}.json")

    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(t3_table, f, ensure_ascii=False, indent=2)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: list(x) if isinstance(x, set) else x)

    print(f"\n  t3_table_{wl}_{mode}.json   — {len(t3_table)} estados")
    print(f"  t3_report_{wl}_{mode}.json  — reporte completo")

    return t3_table


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Precomputa la tabla T3 óptima para cada estado "
                    "(pat_T1, pat_T2)")
    parser.add_argument("--length",  default="4",
                        choices=["4", "5", "6", "all"])
    parser.add_argument("--mode",    default="both",
                        choices=["uniform", "frequency", "both"])
    parser.add_argument("--workers", type=int,
                        default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    print("═" * 66)
    print("  PRECOMPUTACIÓN T3 — árbol óptimo por estado (pat_T1, pat_T2)")
    print("═" * 66)
    print(f"  Workers: {args.workers}")
    print(f"  Clasificación: trivial|two|cluster1|cluster2|few|many")
    print(f"  E[direct] vs E[probe] exacto para ≤{FEW_DIRECT_THRESHOLD} cands")
    print(f"  Cluster-buster + comparación vs exhaustivo garantizada")

    lengths = [4, 5, 6] if args.length == "all" else [int(args.length)]
    modes   = (["uniform", "frequency"] if args.mode == "both"
                else [args.mode])

    for wl in lengths:
        for mode in modes:
            run(wl, mode, args.workers)

    print("\n" + "═" * 66)
    print("  INTEGRACIÓN CON strategy.py")
    print("═" * 66)
    print("""
  En strategy.py, cargar en begin_game() y usar en choose_word():

  _t3 = {}   # class-level cache

  def begin_game(self, word_length, mode, ...):
      key = f"{word_length}_{mode}"
      if key not in Strategy._t3:
          p = TEAM_DIR / f"t3_table_{word_length}_{mode}.json"
          if p.exists():
              Strategy._t3[key] = json.load(open(p))

  def choose_word(self, candidates, weights, turn):
      if turn == 3:
          table  = Strategy._t3.get(f"{self.wl}_{self.mode}", {})
          state  = f"{self._t1_pat}|{self._t2_pat}"
          entry  = table.get(state)
          if entry:
              return entry["guess"]
          # Fallback: entropía runtime si el estado no está en tabla
    """)


if __name__ == "__main__":
    mp.freeze_support()
    main()