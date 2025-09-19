import json

from Dataset_Gen.utils import dict_to_prompt_completion
from utils import make_training_example
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import json, random, os

split_name = "2_only_Start"
filename = f'maze_training_{split_name}.json'

# --- Define sizes and counts ---
size_counts = {
    3: 100,   # 1000 mazes of size 5x5
    4: 200,       # 500 mazes of size 7x7
    5: 300,       # 250 mazes of size 9x9
    #6: 3000,
    #7: 4000,      # 100 mazes of size 11x11
}

all_examples = []
for g, n in size_counts.items():
    for count in range(n):
        m = Maze()
        m.generator = Prims(g, g)
        m.solver = BacktrackingSolver()
        m.generate_monte_carlo(5, 10, 0.5)
        all_examples.append(make_training_example(m))

# --- Save to JSON ---
with open(filename, "w") as f:
    json.dump(all_examples, f, indent=2)

data = all_examples
random.shuffle(data)
# 90% train, 10% valid
split = int(0.9 * len(data))
train_data = data[:split]
valid_data = data[split:]

os.makedirs("../data", exist_ok=True)

def save_jsonl(examples, path):
    with open(path, "w") as fout:
        for ex in examples:
            fout.write(dict_to_prompt_completion(ex, True, False, False))

os.makedirs(f'../data/custom_{split_name}', exist_ok=True)
save_jsonl(train_data, f'../data/custom_{split_name}/train.jsonl')
save_jsonl(valid_data, f'../data/custom_{split_name}/valid.jsonl')

print(f"✅ Wrote {len(train_data)} train and {len(valid_data)} valid examples")
