"""
GENERADOR DE NO-PALABRAS PARA DP — orientado a clusters difíciles
═══════════════════════════════════════════════════════════════════

El problema: nuestras 100 no-palabras actuales son "globales" — diseñadas
para maximizar entropía desde el estado inicial. Son buenas para T1/T2 pero
no sirven para romper clusters difíciles en T3/T4.

Ejemplo del problema:
  Cluster -ATO: gato, pato, mato, dato, rato, hato (6+ palabras)
  Después de T1 (aore) + T2, puede quedar este cluster intacto.
  No-palabra ideal: algo que testee G,P,M,D,R en posición 1 a la vez.
  Nuestra heurística global NUNCA generaría "gpmr" porque g,p,m,r
  no son las letras más frecuentes del vocabulario.

Estrategia de este script:
  1. Detectar todos los clusters del vocabulario (grupos de ≥3 palabras
     que comparten N-1 de N letras en las mismas posiciones)
  2. Para cada cluster, generar no-palabras que maximicen la separación
     de ese cluster específico
  3. Combinar con las no-palabras globales actuales
  4. Exportar pool final para usar en parallel_dp.py

Corre desde: cd ~/ia/wordle/torneo_wordle
Comando: python3 generate_nonwords.py
Output: nonwords_pool_4letter.json (lista de strings)
"""

import csv
import itertools
import json
import math
import re
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
SPANISH_LETTERS = list("abcdefghijklmnñopqrstuvwxyz")


# ══════════════════════════════════════════════════════════════════════════════
# Carga
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


def load_vocab(wl=4):
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
    return words, counts


# ══════════════════════════════════════════════════════════════════════════════
# Detección de clusters
# ══════════════════════════════════════════════════════════════════════════════

def find_clusters(vocab, wl=4, min_size=3):
    """
    Encuentra grupos de palabras que comparten N-1 letras en las mismas
    posiciones (solo difieren en 1 posición).

    Estos son los clusters que causan que el árbol sea profundo.
    """
    # Agrupar por "patrón con posición i enmascarada"
    clusters = []
    seen_clusters = set()

    for mask_pos in range(wl):
        groups = defaultdict(list)
        for w in vocab:
            # Crear clave enmascarando posición mask_pos
            key = w[:mask_pos] + '*' + w[mask_pos+1:]
            groups[key].append(w)

        for key, words in groups.items():
            if len(words) >= min_size:
                frozen = frozenset(words)
                if frozen not in seen_clusters:
                    seen_clusters.add(frozen)
                    # La posición ambigua es mask_pos
                    ambig_letters = [w[mask_pos] for w in words]
                    clusters.append({
                        'pattern': key,
                        'words': sorted(words),
                        'ambig_pos': mask_pos,
                        'ambig_letters': sorted(set(ambig_letters)),
                        'size': len(words),
                    })

    # Ordenar por tamaño descendente (los más difíciles primero)
    clusters.sort(key=lambda x: -x['size'])
    return clusters


def find_deep_clusters(vocab, wl=4, min_size=3):
    """
    Clusters donde difieren 2 o más posiciones pero siguen siendo
    un grupo cohesivo difícil de resolver.

    Agrupa por feedback compartido: si feedback(w1, g) == feedback(w2, g)
    para el opener 'aore', esas palabras quedan en el mismo grupo T1.
    """
    opener = 'aore'
    t1_groups = defaultdict(list)
    for w in vocab:
        t1_groups[feedback(w, opener)].append(w)

    deep = []
    for pat, words in t1_groups.items():
        if len(words) >= min_size:
            deep.append({
                'pattern': ''.join(str(x) for x in pat),
                'words': sorted(words),
                'size': len(words),
            })

    deep.sort(key=lambda x: -x['size'])
    return deep


# ══════════════════════════════════════════════════════════════════════════════
# Generación de no-palabras globales (enfoque actual mejorado)
# ══════════════════════════════════════════════════════════════════════════════

def generate_global_nonwords(vocab, wl=4, n=200):
    """
    No-palabras globales: combinaciones de letras frecuentes.
    Versión ampliada del método actual (200 en lugar de 100).
    """
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
    # Usar top-12 letras para más variedad
    pool = top[:min(wl + 8, len(top))]

    non_words, seen = [], set()
    for r in range(wl, min(len(pool) + 1, wl + 5)):
        for combo in itertools.combinations(pool[:r], wl):
            for perm in itertools.permutations(combo):
                nw = "".join(perm)
                if nw not in seen and nw not in vocab_set:
                    seen.add(nw)
                    non_words.append(nw)
                if len(non_words) >= n:
                    return non_words
    return non_words


# ══════════════════════════════════════════════════════════════════════════════
# Generación de no-palabras cluster-targeted
# ══════════════════════════════════════════════════════════════════════════════

def generate_cluster_nonwords(clusters, vocab, wl=4):
    """
    Para cada cluster, genera no-palabras diseñadas para separar ese cluster.

    Estrategia: si el cluster tiene letras ambiguas {G,P,M,D,R} en posición 0
    y las posiciones 1,2,3 son conocidas ({A,T,O}), necesitamos una no-palabra
    que testee 4 de esas 5 letras ambiguas en posición 0 y use otras letras
    informativas en posiciones 1,2,3 (no las ya conocidas A,T,O porque ya
    las sabemos).

    Para eso construimos strings que:
      - En la posición ambigua: ponen las letras más ambiguas del cluster
      - En otras posiciones: ponen letras que NO aparecen en el cluster
        (para no "desperdiciar" posiciones en info que ya tenemos)
    """
    vocab_set = set(vocab)
    non_words = []
    seen = set()

    for cluster in clusters:
        ambig_pos = cluster['ambig_pos']
        ambig_letters = cluster['ambig_letters']
        words = cluster['words']

        if len(ambig_letters) <= 2:
            continue  # Con 2 opciones ya se resuelve solo

        # Letras fijas del cluster (las que comparten todas las palabras)
        fixed_letters = set()
        for pos in range(wl):
            if pos != ambig_pos:
                letters_at_pos = set(w[pos] for w in words)
                if len(letters_at_pos) == 1:
                    fixed_letters.add(list(letters_at_pos)[0])

        # Letras que NO están en el cluster — útiles para posiciones no-ambiguas
        all_letters_in_cluster = set()
        for w in words:
            all_letters_in_cluster.update(w)
        available_letters = [ch for ch in SPANISH_LETTERS
                             if ch not in all_letters_in_cluster]

        # Construir no-palabra: en posición ambigua poner una letra del cluster,
        # en otras posiciones poner letras fuera del cluster
        # Generamos varias variantes probando cada letra ambigua en ambig_pos
        for primary_letter in ambig_letters[:8]:  # top-8 letras ambiguas
            # Buscar combinación con letras fuera del cluster en otras pos
            for filler_combo in itertools.combinations(available_letters[:10], wl - 1):
                candidate = list(filler_combo)
                candidate.insert(ambig_pos, primary_letter)
                nw = ''.join(candidate)

                # Validar: letras únicas, no es palabra
                if (len(set(nw)) == wl and
                        nw not in seen and
                        nw not in vocab_set):
                    seen.add(nw)
                    non_words.append(nw)
                    break  # Una por letra ambigua es suficiente

        # Adicionalmente: no-palabras que testean MÚLTIPLES letras ambiguas
        # a la vez en diferentes posiciones
        if len(ambig_letters) >= 4:
            for combo in itertools.combinations(ambig_letters, min(4, wl)):
                for perm in itertools.permutations(combo):
                    nw = ''.join(perm)
                    if (len(set(nw)) == wl and
                            nw not in seen and
                            nw not in vocab_set):
                        seen.add(nw)
                        non_words.append(nw)
                        break

    print(f"  No-palabras cluster-targeted: {len(non_words)}")
    return non_words


# ══════════════════════════════════════════════════════════════════════════════
# No-palabras para clusters post-opener (los más relevantes para el DP)
# ══════════════════════════════════════════════════════════════════════════════

def generate_t1_group_nonwords(vocab, wl=4, n_per_group=10):
    """
    Para cada grupo T1 grande (post-opener), genera no-palabras que
    maximizan la separación de ese grupo específico.

    Estas son las no-palabras más relevantes para el DP porque son
    exactamente los estados que el DP necesita resolver eficientemente.
    """
    opener = 'aore'
    vocab_set = set(vocab)
    non_words = []
    seen = set()

    # Grupos T1
    t1_groups = defaultdict(list)
    for w in vocab:
        t1_groups[feedback(w, opener)].append(w)

    # Ordenar por tamaño descendente
    large_groups = [(pat, words) for pat, words in t1_groups.items()
                    if len(words) >= 5]
    large_groups.sort(key=lambda x: -len(x[1]))

    print(f"  Grupos T1 con ≥5 candidatos: {len(large_groups)}")

    for pat, words in large_groups:
        # Identificar qué posiciones son ambiguas en este grupo
        pos_letters = [set(w[i] for w in words) for i in range(wl)]
        ambig_positions = [(i, letters) for i, letters in enumerate(pos_letters)
                           if len(letters) > 1]

        if not ambig_positions:
            continue

        # Para este grupo, las letras más ambiguas son las que más palabras
        # distinguen de un guess
        all_letters_in_group = set()
        for w in words:
            all_letters_in_group.update(w)

        # Letras fuera del grupo (para usar como "fillers informativos")
        outside_letters = [ch for ch in SPANISH_LETTERS
                           if ch not in all_letters_in_group]

        # Generar strings que combinen letras ambiguas de distintas posiciones
        ambig_letter_sets = [letters for _, letters in ambig_positions]

        # Tomar una letra de cada posición ambigua y combinarlas
        count = 0
        for combo in itertools.product(*[sorted(s)[:4]
                                          for s in ambig_letter_sets[:wl]]):
            if len(set(combo)) < len(combo):
                continue  # letras repetidas — saltar
            # Rellenar posiciones faltantes con letras fuera del grupo
            nw_letters = list(combo)
            needed = wl - len(nw_letters)
            for filler in outside_letters:
                if filler not in nw_letters and needed > 0:
                    nw_letters.append(filler)
                    needed -= 1

            if len(nw_letters) != wl:
                continue

            # Probar todas las permutaciones de estas letras
            for perm in itertools.permutations(nw_letters):
                nw = ''.join(perm)
                if (len(set(nw)) == wl and
                        nw not in seen and
                        nw not in vocab_set):
                    seen.add(nw)
                    non_words.append(nw)
                    count += 1
                    break

            if count >= n_per_group:
                break

    print(f"  No-palabras T1-group-targeted: {len(non_words)}")
    return non_words


# ══════════════════════════════════════════════════════════════════════════════
# Evaluación de calidad del pool
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_pool(non_words, vocab, wl=4, n_test=20):
    """
    Muestra cuánta entropía añaden las no-palabras al pool.
    Toma una muestra de grupos difíciles y verifica si las no-palabras
    son útiles para separarlos.
    """
    opener = 'aore'
    t1_groups = defaultdict(list)
    for w in vocab:
        t1_groups[feedback(w, opener)].append(w)

    large_groups = sorted([(words) for words in t1_groups.values()
                           if len(words) >= 8], key=lambda x: -len(x))[:n_test]

    print(f"\n  Evaluando utilidad de no-palabras en {len(large_groups)} grupos difíciles...")

    weights = {w: 1.0/len(vocab) for w in vocab}

    def entropy(guess, candidates, w):
        from collections import defaultdict
        partition = defaultdict(float)
        for cand in candidates:
            pat = feedback(cand, guess)
            partition[pat] += w.get(cand, 0.0)
        total = sum(partition.values())
        if total == 0: return 0.0
        h = 0.0
        for mass in partition.values():
            p = mass / total
            if p > 0: h -= p * math.log2(p)
        return h

    improvements = 0
    for words in large_groups:
        w_local = {word: 1.0/len(words) for word in words}

        # Mejor candidato del vocab
        best_vocab = max(vocab, key=lambda g: entropy(g, words, w_local))
        h_vocab = entropy(best_vocab, words, w_local)

        # Mejor no-palabra
        best_nw = max(non_words, key=lambda g: entropy(g, words, w_local))
        h_nw = entropy(best_nw, words, w_local)

        if h_nw > h_vocab + 0.01:
            improvements += 1
            print(f"    Grupo {[w for w in words[:4]]}... ({len(words)} words)")
            print(f"      Mejor vocab:    '{best_vocab}' H={h_vocab:.4f}")
            print(f"      Mejor no-pal:   '{best_nw}' H={h_nw:.4f} (+{h_nw-h_vocab:.4f})")

    print(f"  No-palabras mejoran {improvements}/{len(large_groups)} grupos difíciles")
    return improvements


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  GENERADOR DE NO-PALABRAS — clusters difíciles 4 letras")
    print("═" * 60)

    vocab, counts = load_vocab(WORD_LENGTH)
    vocab_set = set(vocab)

    print(f"\n  Vocabulario: {len(vocab)} palabras")

    # 1. Detectar clusters
    print(f"\n  Detectando clusters (posición única ambigua)...")
    clusters = find_clusters(vocab, WORD_LENGTH, min_size=3)
    print(f"  Clusters detectados: {len(clusters)}")
    print(f"  Top-10 más grandes:")
    for c in clusters[:10]:
        print(f"    '{c['pattern']}' ({c['size']} palabras): "
              f"{c['words'][:5]}{'...' if len(c['words'])>5 else ''}")
              
    # 2. Clusters post-opener
    print(f"\n  Clusters post-opener 'aore'...")
    deep = find_deep_clusters(vocab, WORD_LENGTH, min_size=5)
    print(f"  Grupos T1 con ≥5 candidatos: {len(deep)}")
    print(f"  Top-10 más grandes:")
    for d in deep[:10]:
        print(f"    pat={d['pattern']} ({d['size']} palabras): "
              f"{d['words'][:4]}{'...' if len(d['words'])>4 else ''}")

    # 3. Generar no-palabras
    print(f"\n  Generando no-palabras...")

    nw_global  = generate_global_nonwords(vocab, WORD_LENGTH, n=200)
    nw_cluster = generate_cluster_nonwords(clusters, vocab, WORD_LENGTH)
    nw_t1      = generate_t1_group_nonwords(vocab, WORD_LENGTH, n_per_group=15)

    # Combinar y deduplicar
    all_nw = []
    seen = set(vocab)
    for nw_list in [nw_global, nw_cluster, nw_t1]:
        for nw in nw_list:
            if nw not in seen:
                seen.add(nw)
                all_nw.append(nw)

    print(f"\n  Pool final de no-palabras:")
    print(f"    Globales:         {len(nw_global)}")
    print(f"    Cluster-targeted: {len(nw_cluster)}")
    print(f"    T1-group:         {len(nw_t1)}")
    print(f"    TOTAL único:      {len(all_nw)}")

    # 4. Evaluar calidad
    evaluate_pool(all_nw, vocab, WORD_LENGTH)

    # 5. Guardar
    out = Path("nonwords_pool_4letter.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_nw, f, ensure_ascii=False)
    print(f"\n  ✓ Guardado en {out} ({len(all_nw)} no-palabras)")

    # Estimar impacto en tiempo del DP
    pool_size = len(vocab) + len(all_nw)
    ratio = pool_size / (len(vocab) + 100)
    print(f"\n  Pool total para DP: {pool_size} ({len(vocab)} vocab + {len(all_nw)} no-pal)")
    print(f"  Impacto en tiempo DP vs pool actual: ~{ratio:.1f}x")
    print(f"\n  SIGUIENTE PASO:")
    print(f"  Actualiza parallel_dp.py para cargar nonwords_pool_4letter.json")
    print(f"  en lugar de generate_non_words(vocab, wl, n=100)")


if __name__ == "__main__":
    main()