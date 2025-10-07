import json
from sklearn.model_selection import train_test_split
from utils import dict_to_prompt_completion, build_prompt, build_target, make_training_example, TASKS
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import json, os

split_name = "test"

filename = f'../data/maze_training_{split_name}.json'
dataset_dir = f'../data/custom_{split_name}'

os.makedirs("../data", exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

# --- Define sizes and counts ---
size_counts = {
    3: 500,   # 1000 mazes of size 5x5
    4: 5000,       # 500 mazes of size 7x7
    5: 7000,       # 250 mazes of size 9x9
    6: 12000,
    7: 15000,      # 100 mazes of size 11x11
}

Maze.set_seed(123)

all_examples = []
for g, n in size_counts.items():
    for count in range(n):
        m = Maze()
        m.generator = Prims(g, g)
        m.solver = BacktrackingSolver()
        if count < n//4:
            m.generate_monte_carlo(5, 10, 0.5)
        else:
            m.generate()
            m.generate_entrances(start_outer=False, end_outer=False)
        m.solve()
        all_examples.append(make_training_example(m, TASKS))

# --- Save to JSON ---
with open(filename, "w") as f:
    json.dump(all_examples, f, indent=2)

data = all_examples

# Extract labels for stratification
labels = [d["maze_size"] for d in data]

# Split indices instead of the dicts directly
indices = list(range(len(data)))

train_idx, valid_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# Map back to dicts
train_data = [data[i] for i in train_idx]
valid_data = [data[i] for i in valid_idx]

# Extract labels for stratification
labels = [d["maze_size"] for d in valid_data]
# Split indices instead of the dicts directly
indices = list(range(len(valid_data)))

valid_idx, test_idx = train_test_split(
    indices,
    test_size=0.33,
    stratify=labels,
    random_state=42
)
valid_data = [data[i] for i in valid_idx]
test_data = [data[i] for i in test_idx]

def save_jsonl(examples, path):
    with open(path, "w") as fout:
        for ex in examples:
            fout.write(dict_to_prompt_completion(ex, target_mode))


save_jsonl(train_data, f'{dataset_dir}/train.jsonl')
save_jsonl(valid_data, f'{dataset_dir}/valid.jsonl')
save_jsonl(test_data, f'{dataset_dir}/test.jsonl')

print(f"Wrote {len(train_data)} train and {len(valid_data)} valid and {len(test_data)} test examples")
