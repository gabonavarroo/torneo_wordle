import math
import time
import multiprocessing
from functools import partial
from lexicon import load_lexicon
from wordle_env import feedback

def calculate_entropy(guess, candidates, probabilities):
    """
    Calcula la entropía ponderada de un guess contra todos los candidatos posibles.
    """
    pattern_probs = {}
    
    # Agrupamos la probabilidad de los candidatos por el patrón de feedback que generan
    for secret in candidates:
        # feedback(secret, guess) nos da una tupla ej: (0, 1, 2, 0, 1)
        patron = feedback(secret, guess)
        prob_secret = probabilities.get(secret, 0)
        
        if patron not in pattern_probs:
            pattern_probs[patron] = 0
        pattern_probs[patron] += prob_secret
        
    # Calculamos la entropía de Shannon ponderada
    entropy = 0
    for p in pattern_probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
            
    return guess, entropy

def find_best_opener(length, mode):
    print(f"\n--- Precomputando Opener para L={length}, Modo={mode} ---")
    start_time = time.time()
    
    # CORRECCIÓN: Usamos kwargs para claridad y extraemos los datos del objeto Lexicon
    lex = load_lexicon(word_length=length, mode=mode)
    words = lex.words
    probabilities = lex.probs
    
    # Preparamos la función parcial para el map de multiprocesamiento
    calc_func = partial(calculate_entropy, candidates=words, probabilities=probabilities)
    
    best_guess = None
    max_entropy = -1.0
    
    # Usamos todos los núcleos de la CPU para paralelizar los millones de cálculos
    num_cores = multiprocessing.cpu_count()
    print(f"Evaluando {len(words)} palabras usando {num_cores} núcleos...")
    
    with multiprocessing.Pool(num_cores) as pool:
        # Mapeamos la función a todas las palabras del vocabulario
        results = pool.map(calc_func, words)
        
    # Encontramos el guess con la máxima entropía
    for guess, entropy in results:
        if entropy > max_entropy:
            max_entropy = entropy
            best_guess = guess
            
    elapsed = time.time() - start_time
    print(f"Mejor opener encontrado: '{best_guess}' con entropía: {max_entropy:.4f} bits")
    print(f"Tiempo de cálculo: {elapsed:.2f} segundos")
    
    return best_guess, max_entropy

if __name__ == '__main__':
    # Los 6 escenarios del torneo
    scenarios = [
        (4, 'uniform'),
        (4, 'frequency'),
        (5, 'uniform'),
        (5, 'frequency'),
        (6, 'uniform'),
        (6, 'frequency')
    ]
    
    best_openers = {}
    
    for L, mode in scenarios:
        best_word, bits = find_best_opener(L, mode)
        best_openers[(L, mode)] = best_word
        
    print("\n" + "="*50)
    print("¡PRECOMPUTACIÓN TERMINADA!")
    print("Copia y pega este diccionario en el método begin_game de tu strategy.py:\n")
    
    print("self.openers = {")
    for (L, mode), word in best_openers.items():
        print(f"    ({L}, '{mode}'): '{word}',")
    print("}")
    print("="*50)