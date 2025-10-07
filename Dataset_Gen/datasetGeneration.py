import json
from sklearn.model_selection import train_test_split
from utils import dict_to_prompt_completion, make_training_example, TASKS, filter_jsonl_by_task_ratio
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import json, os

split_name = "curriculum_1"

filename = f'./data/maze_training_{split_name}.json'
dataset_dir = f'./data/custom_{split_name}'

os.makedirs("./data", exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

# --- Define sizes and counts ---
size_counts = {
    3: 500,   # 1000 mazes of size 5x5
    4: 5000,       # 500 mazes of size 7x7
    5: 7000,       # 250 mazes of size 9x9
    6: 10000,
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
        all_examples.append(make_training_example(m, TASKS, id= f"{g}x{g}_{count}"))

# --- Save to JSON ---
with open(filename, "w") as f:
    json.dump(all_examples, f, indent=2)

data = all_examples

# First split: train vs rest
labels = [d["maze_size"] for d in data]
indices = list(range(len(data)))
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# Second split: valid vs test from the remaining data
temp_data = [data[i] for i in temp_idx]
temp_labels = [d["maze_size"] for d in temp_data]
temp_indices = list(range(len(temp_data)))
valid_idx, test_idx = train_test_split(
    temp_indices,
    test_size=0.33,
    stratify=temp_labels,
    random_state=42
)

# Create final datasets
train_data = [data[i] for i in train_idx]
valid_data = [temp_data[i] for i in valid_idx]
test_data = [temp_data[i] for i in test_idx]

def save_jsonl(examples, path):
    with open(path, "w") as fout:
        for ex in examples:
            fout.write(dict_to_prompt_completion(ex))

# Define task ratios (must sum to 1.0)
task_ratios = {
    "DETECT_START_END": 0.1,
    "AVAILABLE_DIRECTIONS": 0.2,
    "VALID_MOVE": 0.2,
    "OPTIMAL_NEXT_STEP": 0.5
}

# Clean up any existing files first
for file in ['train.jsonl', 'test.jsonl', 'valid.jsonl']:
    file_path = os.path.join(dataset_dir, file)
    if os.path.exists(file_path):
        os.remove(file_path)

# Process each split
splits = {
    'train': train_data,
    'valid': valid_data,
    'test': test_data
}

for split_name, split_data in splits.items():
    # Filter and save directly to final location
    filtered_jsonl = filter_jsonl_by_task_ratio(
        input_data=split_data,
        task_ratios=task_ratios
    )
    
    output_path = f'{dataset_dir}/{split_name}.jsonl'
    with open(output_path, "w") as f:
        f.write(filtered_jsonl)

print(f"Wrote {len(train_data)} train and {len(valid_data)} valid and {len(test_data)} test examples")
