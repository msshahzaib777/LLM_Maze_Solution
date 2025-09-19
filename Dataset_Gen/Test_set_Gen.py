import os
from Dataset_Gen.utils import dict_to_prompt_completion
from utils import make_training_example
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver

# --- Define complexity & size counts ---
complexity_counts = {
    3: {1: 200, 2: 200, 3: 200},
    4: {1: 200, 2: 200, 3: 200},
    5: {1: 300, 2: 300, 3: 300, 4: 300},
    6: {1: 300, 2: 300, 3: 300, 4: 300},
    7: {1: 400, 2: 400, 3: 400, 4: 400, 5: 400},
}


# --- Save JSONL ---
def save_jsonl(examples, path):
    with open(path, "w") as fout:
        for ex in examples:
            fout.write(dict_to_prompt_completion(ex, True, False, False))

# --- Main ---
output_dir = "../data/test_data"
os.makedirs(output_dir, exist_ok=True)

for size, complexities in complexity_counts.items():
    for complexity, n in complexities.items():
        examples = []
        for _ in range(n):
            m = Maze()
            m.generator = Prims(size, size)
            m.solver = BacktrackingSolver()

            # complexity tuning via Monte Carlo parameters
            m.generate_monte_carlo(20, 10, complexity * 0.2)

            examples.append(make_training_example(m))

        filename = f"maze_test_size{size}_complex{complexity}.jsonl"
        save_jsonl(examples, os.path.join(output_dir, filename))
        print(f"✅ Saved {len(examples)} test examples to {filename}")
