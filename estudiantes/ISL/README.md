# Team Template

Copy this directory to create your team workspace:

```bash
cp -r estudiantes/_template estudiantes/your_team_name
```

## Directory Structure

```
estudiantes/your_team_name/
    strategy.py          # YOUR STRATEGY (this is what gets submitted)
    results/             # Auto-created: experiment and tournament outputs
    ...                  # Anything else you want (notebooks, scripts, data)
```

## Quick Start

1. **Edit** `strategy.py` — change the class name, the `name` property, and implement `guess()`.

2. **Test** your strategy:
   ```bash
   python experiment.py --strategy "YourName_teamname" --num-games 10 --verbose
   ```

3. **Compare** against benchmarks (Random, MaxProb, Entropy):
   ```bash
   python tournament.py --team your_team_name --num-games 20
   ```

4. **Run specific configurations:**
   ```bash
   # 6-letter words, frequency mode
   python experiment.py --strategy "YourName_teamname" --length 6 --mode frequency --num-games 20 --verbose

   # Full local tournament (all configs)
   python tournament.py --team your_team_name --official --num-games 10
   ```

## Rules

- Your strategy must work for **all** word lengths (4, 5, 6) and **both** modes (uniform, frequency).
- **5 seconds** max per game (one secret word, up to 6 guesses).
- **1 CPU core** during tournament.
- Only `numpy` + standard library allowed (no extra dependencies).
- No machine learning or reinforcement learning. Use goal-based or utility-based approaches. Simulations are allowed.
- The `name` property must be unique: `"StrategyName_teamname"`.

## Useful Utilities

```python
from wordle_env import feedback, filter_candidates
from lexicon import load_lexicon

# Compute feedback for a guess against a secret
fb = feedback("canto", "arcos")  # -> (1, 0, 1, 1, 0)

# Filter candidates by feedback
remaining = filter_candidates(word_list, "arcos", (1, 0, 1, 1, 0))

# Load word list with probabilities
lex = load_lexicon(word_length=5, mode="frequency")
print(lex.words[:5], lex.probs)
```
