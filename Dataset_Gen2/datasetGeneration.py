import json
from sklearn.model_selection import train_test_split
try:
    from .utils import (
        dict_to_prompt_completion,
        make_training_example,
        TASKS,
        suggest_optimal_max_seq_length,
        open_jsonl,
        save_jsonl
    )
except ImportError:
    from utils import (
        dict_to_prompt_completion,
        make_training_example,
        TASKS,
        suggest_optimal_max_seq_length,
        open_jsonl,
        save_jsonl
    )
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import os
import sys
from multiprocessing import Pool, cpu_count
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import DATA_DIR

def generate_single_maze(args):
    """Worker function for parallel maze generation"""
    g, count, monte_carlo_params, use_monte_carlo = args
    
    m = Maze()
    m.generator = Prims(g, g)
    m.solver = BacktrackingSolver()
    
    if use_monte_carlo:
        m.generate_monte_carlo(monte_carlo_params["repeat"], monte_carlo_params["entrances"], monte_carlo_params['difficulty'])
    else:
        m.generate()
        m.generate_entrances(start_outer=False, end_outer=False)
    
    m.solve()
    return make_training_example(m, TASKS, id=f"{g}x{g}_{count}")

def generate_maze_examples(config):
    """Generate maze examples based on size counts using multiprocessing"""
    Maze.set_seed(123)
    worker_args = [(g, c, config["monte_carlo_params"], c < n//4) 
                   for g, n in config['maze_sizes'].items() for c in range(n)]
    
    print(f"Generating {len(worker_args)} mazes using {cpu_count()} cores...")
    start = time.time()
    with Pool(processes=min(cpu_count(), 8)) as pool:
        result = pool.map(generate_single_maze, worker_args)
    print(f"Generation completed in {time.time() - start:.2f} seconds")
    return result

def data_splitter(data, splits, split_ratio=None):
    """Split data into sets based on provided split names"""
    if len(splits) < 2:
        raise ValueError("At least two splits are required")
    
    result, remaining_idx = {}, list(range(len(data)))
    remaining_labels = [d["maze_size"] for d in data]
    
    for i, split_name in enumerate(splits[:-1]):
        test_size = split_ratio.get("test" if i == 0 else "valid", 0.3 if i == 0 else 0.5)
        split_idx, remaining_idx = train_test_split(remaining_idx, test_size=test_size, stratify=remaining_labels, random_state=42)
        result[split_name] = [data[i] for i in split_idx]
        remaining_labels = [data[i]["maze_size"] for i in remaining_idx]
    
    result[splits[-1]] = [data[i] for i in remaining_idx]
    return result

def main(CONFIG=None):
    start = time.time()
    filename = f'{DATA_DIR}/maze_training_{CONFIG["dataset_name"]}.json'
    dataset_dir = f'{DATA_DIR}/{CONFIG["dataset_name"]}'
    os.makedirs(dataset_dir, exist_ok=True)

    # Generate or load maze examples
    if not CONFIG['skip_generation'] and not os.path.exists(filename):
        print("Generating maze examples with environment-agent interaction...")
        all_examples = generate_maze_examples(CONFIG)
        print(f"Saving {len(all_examples)} examples to {filename}")
        with open(filename, "w") as f:
            json.dump(all_examples, f, indent=2)
    else:
        print(f"Loading existing maze examples from {filename}...")
        with open(filename) as f:
            all_examples = json.load(f)
    
    print("Creating train/valid/test splits...")
    splits_data = data_splitter(all_examples, CONFIG["splits"], CONFIG['split_params'])
    
    # Save splits and count samples
    for split_name, split_data in splits_data.items():
        output_path = f'{dataset_dir}/{split_name}.jsonl'
        print(f"Saving {split_name}.jsonl with {len(split_data)} examples...")
        save_jsonl(split_data, output_path, mapper=dict_to_prompt_completion)
        with open(output_path) as f:
            print(f"  -> {sum(1 for line in f if line.strip())} training samples generated")

    print("Computing sequence length summary...")
    summary = {s: suggest_optimal_max_seq_length(f'{dataset_dir}/{s}.jsonl') for s in CONFIG['splits']}
    with open(f'{dataset_dir}/sequence_length_summary.json', "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Total time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    CONFIG = {
        'random_seed': 123,
        'monte_carlo_params': {
            'repeat': 3,
            'entrances': 5,
            'difficulty': 0.8
        },
        'maze_sizes': {
            3: 0,
            4: 0,
            5: 10947,  # Start with smaller dataset for testing
            6: 0,
            7: 0,
        },
        'splits': ['train', 'valid', 'test'],
        'split_params': {
            'test': 0.10,
            'valid': 0.50,
            'random_state': 42
        },
        'skip_generation': False,
        'dataset_name': 'env_agent_v1',
    }
    print(f"Starting environment-agent dataset generation with {sum(CONFIG['maze_sizes'].values())} total mazes...")
    main(CONFIG)
