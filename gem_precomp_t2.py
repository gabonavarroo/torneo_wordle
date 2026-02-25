import math
import time
import multiprocessing
from lexicon import load_lexicon
from wordle_env import feedback

# Los openers que obtuvimos del primer script
OPENERS = {
    (4, 'uniform'): 'roia',
    (4, 'frequency'): 'cora',
    (5, 'uniform'): 'careo',
    (5, 'frequency'): 'careo',
    (6, 'uniform'): 'careto',
    (6, 'frequency'): 'cerito',
}

def process_branch(args):
    """
    Evalúa una rama del árbol (un patrón de feedback específico).
    Devuelve la tupla: (patrón, mejor_siguiente_intento)
    """
    pattern, candidates, full_vocab, probabilities = args
    
    # CASO BASE: Si quedan 2 candidatos o menos, simplemente adivinamos el más probable.
    # No tiene sentido explorar más, ¡es momento de intentar ganar!
    if len(candidates) <= 2:
        best_cand = max(candidates, key=lambda w: probabilities.get(w, 0))
        return pattern, best_cand

    # Si quedan muchos, buscamos la palabra que maximice la entropía
    max_score = -1.0
    best_guess = candidates[0] # Default
    
    for guess in full_vocab:
        pattern_probs = {}
        
        # Agrupamos los candidatos restantes según el feedback que este guess generaría
        for secret in candidates:
            f = feedback(secret, guess)
            p = probabilities.get(secret, 0)
            pattern_probs[f] = pattern_probs.get(f, 0) + p
            
        # Calculamos la entropía
        ent = 0
        for p in pattern_probs.values():
            if p > 0:
                ent -= p * math.log2(p)
                
        # Tie-breaker (desempate): Si dos palabras dan la misma información, 
        # preferimos la que es un candidato válido (nos da la oportunidad de ganar en este turno).
        is_cand = 1 if guess in candidates else 0
        score = ent + (is_cand * 0.0001)
        
        if score > max_score:
            max_score = score
            best_guess = guess
            
    return pattern, best_guess


def build_turn2_tree(length, mode, opener):
    print(f"\n--- Construyendo Árbol Turno 2 para L={length}, Modo={mode}, Opener='{opener}' ---")
    start_time = time.time()
    
    lex = load_lexicon(word_length=length, mode=mode)
    full_vocab = lex.words
    probabilities = lex.probs
    
    # 1. Simular qué pasaría con cada posible palabra secreta
    # y agruparlas por el feedback que el opener genera.
    branches = {}
    for secret in full_vocab:
        pat = feedback(secret, opener)
        if pat not in branches:
            branches[pat] = []
        branches[pat].append(secret)
        
    print(f"El opener divide el espacio en {len(branches)} ramas (patrones posibles).")
    
    # 2. Preparar los datos para el multiprocesamiento
    tasks = []
    for pat, cands in branches.items():
        tasks.append((pat, cands, full_vocab, probabilities))
        
    # 3. Procesar las ramas en paralelo
    num_cores = multiprocessing.cpu_count()
    print(f"Evaluando ramas en {num_cores} núcleos...")
    
    tree_branch = {}
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_branch, tasks)
        
    for pat, best_guess in results:
        tree_branch[pat] = best_guess
        
    elapsed = time.time() - start_time
    print(f"Terminado en {elapsed:.2f} segundos.")
    
    return tree_branch

if __name__ == '__main__':
    final_tree = {}
    
    for (L, mode), opener in OPENERS.items():
        branch_tree = build_turn2_tree(L, mode, opener)
        final_tree[(L, mode)] = branch_tree
        
    # Guardar el diccionario gigantesco en un archivo Python
    output_filename = "gt2_precomp.py"
    print(f"\nGuardando resultados en {output_filename}...")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("# Este archivo fue generado automáticamente.\n")
        f.write("# Cópialo y pégalo (o impórtalo) en tu strategy.py\n\n")
        f.write("TURN_2_TREE = {\n")
        for (L, mode), branch in final_tree.items():
            f.write(f"    ({L}, '{mode}'): {{\n")
            for pat, guess in branch.items():
                f.write(f"        {pat}: '{guess}',\n")
            f.write("    },\n")
        f.write("}\n")
        
    print(f"¡Listo! Revisa el archivo {output_filename}.")