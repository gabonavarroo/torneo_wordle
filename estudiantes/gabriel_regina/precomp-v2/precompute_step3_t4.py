"""
PASO 3: Precomputar tabla T4 para 4 letras (ambos modos).

Con solo 1,853 palabras y T3 pequeña, el árbol T4 es manejable.
El objetivo es eliminar el fallback dinámico para casi todos los casos
de 4 letras, reduciendo el mean de ~4.6 hacia ~3.8-4.0.

Tiempo estimado: 5-15 minutos (pocos estados T3 generan T4 no-triviales)

Corre desde: cd ~/ia/wordle/torneo_wordle
Comando: python3 precompute_step3_t4.py

Output: precomputed_t4_4letter.json
"""

import csv
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

# ── Openers y tablas T2/T3 del paso 2 ────────────────────────────────────────
# Pegar aquí los valores de precomputed_tables_raw.py
OPENER_4 = 'aore'

T2_4_UNIFORM = {'0000': 'luis', '0001': 'sien', '0002': 'tupi', '0010': 'crin', '0011': 'neis', '0012': 'tuis', '0020': 'buri', '0021': 'heri', '0022': 'sumi', '0100': 'tilo', '0101': 'seco', '0102': 'cien', '0110': 'crio', '0111': 'cedo', '0112': 'croe', '0120': 'tupi', '0121': 'cent', '0122': 'obre', '0200': 'timo', '0201': 'loen', '0202': 'cent', '0210': 'lito', '0211': 'moer', '0212': 'bloc', '0220': 'almo', '0222': 'core', '1000': 'luci', '1001': 'cent', '1002': 'clip', '1010': 'iran', '1011': 'ruar', '1012': 'rafe', '1020': 'tupi', '1021': 'clip', '1022': 'dare', '1100': 'vals', '1101': 'olea', '1110': 'oral', '1120': 'baca', '1200': 'clan', '1210': 'dial', '1220': 'chad', '2000': 'asan', '2001': 'atas', '2002': 'unci', '2010': 'asad', '2011': 'afer', '2012': 'clip', '2020': 'abri', '2022': 'abre', '2100': 'lito', '2101': 'afeo', '2102': 'aloe', '2110': 'amor', '2120': 'abro', '2200': 'aojo'}
T2_4_FREQ   = {'0000': 'luis', '0001': 'sien', '0002': 'tupi', '0010': 'huir', '0011': 'tris', '0012': 'crin', '0020': 'buri', '0021': 'peru', '0022': 'medi', '0100': 'plus', '0101': 'leso', '0102': 'once', '0110': 'fito', '0111': 'reno', '0112': 'orbe', '0120': 'tupi', '0121': 'cepa', '0122': 'obre', '0200': 'cent', '0201': 'soez', '0202': 'cent', '0210': 'moto', '0211': 'roer', '0212': 'rose', '0220': 'filo', '0222': 'more', '1000': 'luci', '1001': 'cent', '1002': 'cent', '1010': 'iran', '1011': 'teca', '1012': 'trae', '1020': 'cadi', '1021': 'vals', '1022': 'hare', '1100': 'vals', '1101': 'olea', '1110': 'oral', '1120': 'capa', '1200': 'clan', '1210': 'clip', '1220': 'chad', '2000': 'unis', '2001': 'ases', '2002': 'tune', '2010': 'arma', '2011': 'area', '2012': 'acto', '2020': 'abri', '2022': 'aire', '2100': 'luto', '2101': 'aseo', '2102': 'aloe', '2110': 'amor', '2120': 'agro', '2200': 'aojo'}

T3_4_UNIFORM = {'0000|luis|0010': 'cinc', '0000|luis|0222': 'cuis', '0000|luis|1010': 'film', '0000|luis|0220': 'quid', '0000|luis|0211': 'sufi', '0001|sien|0110': 'cedi', '0001|sien|0222': 'bien', '0001|sien|0022': 'peen', '0001|sien|0020': 'juez', '0001|sien|1020': 'pues', '0001|sien|0220': 'fiel', '0002|tupi|0000': 'cede', '0002|tupi|0001': 'dime', '0002|tupi|1001': 'bite', '0002|tupi|0200': 'mude', '0002|tupi|0220': 'cupe', '0002|tupi|1000': 'jete', '0002|tupi|1200': 'mute', '0002|tupi|0010': 'peje', '0002|tupi|0011': 'pide', '0002|tupi|0210': 'pude', '0002|tupi|2000': 'tefe', '0002|tupi|2001': 'tibe', '0002|tupi|2200': 'tune', '0010|crin|0220': 'gris', '0010|crin|0120': 'fuir', '0011|neis|0110': 'crei', '0011|neis|0100': 'frey', '0011|neis|0200': 'leer', '0012|tuis|0000': 'rede', '0012|tuis|0010': 'rice', '0012|tuis|0100': 'urbe', '0022|sumi|0200': 'cure', '0022|sumi|0001': 'dire', '0100|tilo|0202': 'dimo', '0100|tilo|1202': 'bito', '0100|tilo|0002': 'cuso', '0100|tilo|0022': 'bulo', '0100|tilo|0102': 'ubio', '0100|tilo|1002': 'cuto', '0100|tilo|0222': 'dilo', '0100|tilo|0212': 'ligo', '0100|tilo|0012': 'ludo', '0100|tilo|0001': 'obus', '0100|tilo|2202': 'tifo', '0100|tilo|2002': 'tubo', '0101|seco|0202': 'ledo', '0101|seco|0222': 'beco', '0101|seco|1202': 'beso', '0101|seco|0212': 'cebo', '0101|seco|0102': 'oleo', '0101|seco|0201': 'leon', '0101|seco|2202': 'sebo', '0102|cien|0010': 'oboe', '0110|crio|0222': 'brio', '0110|crio|0201': 'troj', '0110|crio|0202': 'orto', '0110|crio|0112': 'rijo', '0110|crio|0102': 'rubo', '0111|cedo|0102': 'breo', '0111|cedo|0202': 'rejo', '0112|croe|0212': 'orbe', '0120|tupi|0200': 'curo', '0120|tupi|0001': 'giro', '0121|cent|0200': 'mero', '0200|timo|0002': 'sobo', '0200|timo|0101': 'cosi', '0200|timo|1002': 'boto', '0200|timo|0022': 'como', '0200|timo|0001': 'golf', '0200|timo|0012': 'moco', '0200|timo|0111': 'moji', '0200|timo|2002': 'toco', '0202|cent|0100': 'sope', '0202|cent|0101': 'lote', '0202|cent|2100': 'coce', '0202|cent|0120': 'done', '0210|lito|0002': 'robo', '0212|bloc|0010': 'rode', '0220|almo|0002': 'boro', '1000|luci|0000': 'yang', '1000|luci|0002': 'bati', '1000|luci|0020': 'baca', '1000|luci|1000': 'glas', '1000|luci|0001': 'tina', '1000|luci|0200': 'duma', '1000|luci|0010': 'cada', '1000|luci|0011': 'cias', '1000|luci|1010': 'clac', '1000|luci|0210': 'cuan', '1000|luci|1001': 'dial', '1000|luci|1200': 'dual', '1000|luci|2000': 'lada', '1000|luci|2001': 'lias', '1000|luci|2200': 'luda', '1000|luci|1002': 'mali', '1000|luci|0100': 'unta', '1001|cent|0200': 'leda', '1001|cent|0201': 'jeta', '1001|cent|2200': 'ceba', '1001|cent|0100': 'edad', '1001|cent|0210': 'vean', '1001|cent|0220': 'mena', '1002|clip|0000': 'taje', '1002|clip|0100': 'late', '1002|clip|2000': 'cabe', '1002|clip|1000': 'hace', '1002|clip|0001': 'pate', '1010|iran|1120': 'ciar', '1010|iran|0220': 'frac', '1010|iran|1210': 'cria', '1010|iran|0120': 'upar', '1010|iran|0110': 'raba', '1010|iran|1110': 'raiz', '1011|ruar|2010': 'reda', '1012|rafe|2202': 'raje', '1020|tupi|0000': 'cara', '1020|tupi|0200': 'cura', '1020|tupi|0001': 'dira', '1021|clip|0000': 'jera', '1022|dare|0222': 'hare', '1100|vals|0200': 'gato', '1100|vals|0220': 'balo', '1100|vals|0201': 'caso', '1100|vals|0210': 'lado', '1100|vals|0100': 'onda', '1100|vals|2200': 'vaco', '1110|oral|2120': 'osar', '1110|oral|1110': 'rabo', '1120|baca|0200': 'faro', '1200|clan|0010': 'sota', '1200|clan|1010': 'boca', '1200|clan|0110': 'bola', '1200|clan|0011': 'tona', '1200|clan|2010': 'coba', '1210|dial|0010': 'roca', '1220|chad|0010': 'jora', '2000|asan|2020': 'atad', '2000|asan|2022': 'adan', '2000|asan|2010': 'aula', '2000|asan|2000': 'agil', '2000|asan|2011': 'anca', '2000|asan|2210': 'asia', '2002|unci|0000': 'afee', '2002|unci|0200': 'ande', '2010|asad|2020': 'ajar', '2010|asad|2010': 'arca', '2012|clip|0000': 'arde', '2022|abre|2022': 'acre', '2100|lito|0022': 'acto', '2100|lito|0102': 'adio', '2100|lito|0002': 'asno', '2100|lito|1002': 'albo', '2100|lito|0001': 'anon', '2101|afeo|2022': 'apeo', '2110|amor|2011': 'arco'}
T3_4_FREQ   = {'0000|luis|0010': 'zinc', '0000|luis|0222': 'huis', '0000|luis|1010': 'film', '0000|luis|0220': 'quin', '0000|luis|0211': 'sufi', '0001|sien|0110': 'pedi', '0001|sien|0222': 'bien', '0001|sien|0022': 'leen', '0001|sien|0020': 'juez', '0001|sien|1020': 'pues', '0001|sien|0220': 'piel', '0002|tupi|0000': 'cede', '0002|tupi|0001': 'dice', '0002|tupi|1001': 'cite', '0002|tupi|0200': 'sume', '0002|tupi|0220': 'supe', '0002|tupi|1000': 'mete', '0002|tupi|1200': 'yute', '0002|tupi|0010': 'pese', '0002|tupi|0011': 'pide', '0002|tupi|0210': 'pude', '0002|tupi|2000': 'teme', '0002|tupi|2001': 'time', '0002|tupi|2200': 'tuve', '0010|huir|0021': 'gris', '0010|huir|0222': 'muir', '0011|tris|0200': 'grey', '0011|tris|0100': 'leer', '0011|tris|0110': 'rien', '0012|crin|0200': 'urge', '0012|crin|0100': 'rule', '0012|crin|0110': 'rige', '0022|medi|0100': 'jure', '0022|medi|0101': 'tire', '0100|plus|0000': 'tino', '0100|plus|0200': 'olmo', '0100|plus|0010': 'tubo', '0100|plus|0110': 'lujo', '0100|plus|0011': 'sumo', '0100|plus|1000': 'tipo', '0100|plus|0100': 'hilo', '0100|plus|0001': 'sido', '0100|plus|2000': 'pido', '0100|plus|2010': 'pudo', '0101|leso|0202': 'neto', '0101|leso|0222': 'peso', '0101|leso|1202': 'pelo', '0101|leso|0102': 'echo', '0101|leso|0111': 'esos', '0101|leso|1102': 'ello', '0101|leso|2202': 'leyo', '0101|leso|0201': 'peon', '0101|leso|0211': 'seos', '0101|leso|0212': 'seno', '0102|once|2002': 'opte', '0110|fito|0001': 'olor', '0110|fito|0002': 'ruso', '0110|fito|0202': 'rico', '0110|fito|0011': 'troy', '0111|reno|1102': 'creo', '0111|reno|2202': 'remo', '0112|orbe|2202': 'orne', '0120|tupi|0200': 'duro', '0120|tupi|0001': 'giro', '0121|cepa|0200': 'mero', '0200|cent|0000': 'solo', '0200|cent|0020': 'kong', '0200|cent|0001': 'tomo', '0200|cent|2000': 'como', '0200|cent|1000': 'poco', '0200|cent|0010': 'nomo', '0202|cent|0100': 'dope', '0202|cent|0101': 'tome', '0202|cent|2100': 'cole', '0202|cent|0120': 'pone', '0210|moto|0202': 'robo', '0212|rose|2202': 'roce', '0220|filo|0002': 'toro', '1000|luci|0000': 'nata', '1000|luci|0002': 'bati', '1000|luci|0020': 'saca', '1000|luci|1000': 'sala', '1000|luci|0001': 'dina', '1000|luci|0200': 'duma', '1000|luci|0010': 'casa', '1000|luci|0011': 'cita', '1000|luci|1010': 'clan', '1000|luci|0210': 'cuba', '1000|luci|1001': 'fila', '1000|luci|1200': 'sula', '1000|luci|2000': 'lama', '1000|luci|2001': 'lisa', '1000|luci|2200': 'luna', '1000|luci|1002': 'mali', '1000|luci|0100': 'usan', '1001|cent|0200': 'lesa', '1001|cent|0201': 'meta', '1001|cent|2200': 'cepa', '1001|cent|0100': 'edad', '1001|cent|0210': 'vean', '1001|cent|0220': 'pena', '1002|cent|0100': 'sale', '1002|cent|0101': 'bate', '1002|cent|2100': 'cafe', '1002|cent|0120': 'gane', '1002|cent|1100': 'hace', '1010|iran|1120': 'liar', '1010|iran|0220': 'fray', '1010|iran|1210': 'fria', '1010|iran|0120': 'usar', '1010|iran|0110': 'rama', '1010|iran|1110': 'raiz', '1011|teca|0101': 'eral', '1012|trae|0112': 'rape', '1020|cadi|0101': 'gira', '1020|cadi|0200': 'para', '1020|cadi|0100': 'pura', '1021|vals|0100': 'mera', '1022|hare|0222': 'pare', '1100|vals|0200': 'taco', '1100|vals|0220': 'malo', '1100|vals|0201': 'caso', '1100|vals|0210': 'lado', '1100|vals|0100': 'onza', '1100|vals|2200': 'vano', '1110|oral|2120': 'otar', '1110|oral|1110': 'rato', '1120|capa|0200': 'raro', '1200|clan|0010': 'sota', '1200|clan|1010': 'boca', '1200|clan|0110': 'mola', '1200|clan|0011': 'nota', '1200|clan|2010': 'copa', '1210|clip|0000': 'roma', '1220|chad|0010': 'mora', '2000|unis|0000': 'abad', '2000|unis|0100': 'afan', '2000|unis|0020': 'amia', '2000|unis|1000': 'azul', '2000|unis|0022': 'asis', '2000|unis|0002': 'amas', '2000|unis|0200': 'anda', '2000|unis|0001': 'asta', '2001|ases|2020': 'amen', '2002|tune|0002': 'aspe', '2010|arma|2100': 'asir', '2010|arma|2101': 'azar', '2010|arma|2201': 'aras', '2010|arma|2202': 'arca', '2012|acto|2000': 'arde', '2022|aire|2022': 'abre', '2100|luto|0002': 'ando', '2100|luto|1002': 'algo', '2100|luto|0001': 'amos', '2101|aseo|2022': 'ateo', '2110|amor|2011': 'arco'}


def _strip_accents(text):
    import unicodedata
    result = []
    for ch in text:
        if ch == "ñ":
            result.append("ñ")
        else:
            decomposed = unicodedata.normalize("NFD", ch)
            result.append("".join(c for c in decomposed if unicodedata.category(c) != "Mn"))
    return "".join(result)


def load_vocab_4():
    csv_path = Path("data/spanish_4letter.csv")
    pattern = re.compile(r"^[a-zñ]{4}$")
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


def f_hat(n):
    if n <= 1: return 1.0
    if n == 2: return 1.5
    if n == 3: return 2.0
    return max(1.0, math.log2(n) * 0.5 + 0.8)


def normalize(candidates, base_w):
    raw = {w: base_w.get(w, 1e-10) for w in candidates}
    total = sum(raw.values())
    if total == 0:
        return {w: 1.0/len(candidates) for w in candidates}
    return {w: v/total for w, v in raw.items()}


def expected_score(guess_word, candidates, weights, wl=4):
    win_pat = tuple([2] * wl)
    partition = defaultdict(list)
    for w in candidates:
        partition[feedback(w, guess_word)].append(w)
    total_w = sum(weights.get(w, 0.0) for w in candidates) or 1.0
    score = 0.0
    for pat, group in partition.items():
        p_f = sum(weights.get(w, 0.0) for w in group) / total_w
        score += p_f * (1.0 if pat == win_pat else 1.0 + f_hat(len(group)))
    return score


def best_from_full_pool(candidates, base_w, full_vocab):
    """Siempre evalúa el vocabulario completo — esto es lo que diferencia
    un T4 precomputado de calidad del fallback que solo ve candidatos."""
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        w = normalize(candidates, base_w)
        return max(candidates, key=lambda x: w.get(x, 0.0))
    w = normalize(candidates, base_w)
    best, best_s = candidates[0], float('inf')
    for g in full_vocab:  # vocabulario COMPLETO
        s = expected_score(g, candidates, w)
        if s < best_s:
            best_s = s
            best = g
    return best


def compute_t4(opener, t2_table, t3_table, vocab, base_w, mode_label):
    """
    Para cada rama que llega a T4 con >2 candidatos, calcula el mejor T4.
    Clave: "{pat1}|{g2}|{pat2}|{g3}|{pat3}" → best_t4_guess
    """
    print(f"\n  T4 {mode_label}...")
    win_pat = (2, 2, 2, 2)
    t4 = {}
    t0 = time.monotonic()

    # Reconstruir T1 groups
    t1_groups = defaultdict(list)
    for w in vocab:
        t1_groups[feedback(w, opener)].append(w)

    nodes_checked = 0
    for pat1, cands1 in sorted(t1_groups.items()):
        if tuple(pat1) == win_pat or len(cands1) <= 2:
            continue
        pat1_str = ''.join(str(x) for x in pat1)
        g2 = t2_table.get(pat1_str)
        if not g2:
            continue

        t2_groups = defaultdict(list)
        for w in cands1:
            t2_groups[feedback(w, g2)].append(w)

        for pat2, cands2 in t2_groups.items():
            if tuple(pat2) == win_pat or len(cands2) <= 2:
                continue
            pat2_str = ''.join(str(x) for x in pat2)
            t3_key = f"{pat1_str}|{g2}|{pat2_str}"
            g3 = t3_table.get(t3_key)
            if not g3:
                continue

            t3_groups = defaultdict(list)
            for w in cands2:
                t3_groups[feedback(w, g3)].append(w)

            for pat3, cands3 in t3_groups.items():
                if tuple(pat3) == win_pat or len(cands3) <= 2:
                    continue
                pat3_str = ''.join(str(x) for x in pat3)
                t4_key = f"{pat1_str}|{g2}|{pat2_str}|{g3}|{pat3_str}"

                # Evaluar vocabulario completo para T4
                best = best_from_full_pool(cands3, base_w, vocab)
                t4[t4_key] = best
                nodes_checked += 1

    elapsed = time.monotonic() - t0
    print(f"  ✓ T4 {mode_label}: {len(t4)} entradas en {elapsed:.1f}s")
    return t4


def main():
    print("=" * 60)
    print("PASO 3: Tabla T4 para 4 letras")
    print("=" * 60)

    vocab, weights_u, weights_f = load_vocab_4()
    print(f"Vocabulario: {len(vocab)} palabras")

    results = {}
    for mode, base_w, t2, t3 in [
        ("uniform",   weights_u, T2_4_UNIFORM, T3_4_UNIFORM),
        ("frequency", weights_f, T2_4_FREQ,   T3_4_FREQ),
    ]:
        t4 = compute_t4(OPENER_4, t2, t3, vocab, base_w, mode)
        results[f"4_{mode}"] = t4

    # Guardar
    with open("precomputed_t4_4letter.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    print("\n✓ Guardado en precomputed_t4_4letter.json")

    # Imprimir para copiar
    print("\n" + "="*60)
    print("Copia esto en strategy.py (reemplaza las tablas T4):")
    print("="*60)
    print(f"_T4_4_UNIFORM    = {repr(results['4_uniform'])}")
    print(f"_T4_4_FREQUENCY  = {repr(results['4_frequency'])}")
    print(f"\nEntradas T4 uniform: {len(results['4_uniform'])}")
    print(f"Entradas T4 frequency: {len(results['4_frequency'])}")


if __name__ == "__main__":
    main()