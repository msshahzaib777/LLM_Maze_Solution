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
import argparse
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
    
    # Prepare arguments for parallel processing
    worker_args = []
    for g, n in config['maze_sizes'].items():
        for count in range(n):
            use_monte_carlo = count < n//4  # 25% use monte carlo
            worker_args.append((g, count, config["monte_carlo_params"], use_monte_carlo))
    
    print(f"Generating {len(worker_args)} mazes using {cpu_count()} cores...")
    start_time = time.time()
    
    # Use multiprocessing to generate mazes in parallel
    with Pool(processes=min(cpu_count(), 8)) as pool:  # Limit to 8 processes to avoid memory issues
        all_examples = pool.map(generate_single_maze, worker_args)
    
    print(f"Generation completed in {time.time() - start_time:.2f} seconds")
    return all_examples

def data_splitter(data, splits, split_ratio=None):
    """Split data into sets based on provided split names"""
    if len(splits) < 2:
        raise ValueError("At least two splits are required")
    
    labels = [d["maze_size"] for d in data]
    indices = list(range(len(data)))
    result = {}
    
    # Handle the first split
    remaining_idx = indices
    remaining_data = data
    remaining_labels = labels
    
    # Split data sequentially based on split names
    for i in range(len(splits) - 1):
        split_name = splits[i]
        
        # Set fixed split sizes: 70% train, 15% valid, 15% test
        if i == 0:  # first split (train)
            test_size = split_ratio.get("test", 0.3)  # keep 70%, split off 30%
        else:  # second split (valid)
            test_size = split_ratio.get("valid", 0.5)  # split remaining 30% equally

        split_idx, remaining_idx = train_test_split(
            remaining_idx, 
            test_size=test_size,
            stratify=remaining_labels,
            random_state=42
        )
        
        result[split_name] = [data[i] for i in split_idx]
        remaining_data = [data[i] for i in remaining_idx]
        remaining_labels = [d["maze_size"] for d in remaining_data]
    
    # Last split gets the remaining data
    result[splits[-1]] = remaining_data
    
    # Return splits in the same order as input splits list
    return result

def process_splits_with_ratios(dataset_dir, task_ratios, splits, seed=42):
    import math, random, json
    rng = random.Random(seed)
    if abs(sum(task_ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    for split_name, raw_data in splits.items():
        print(f"Processing {split_name} split with {len(raw_data)} examples...")
        start_time = time.time()
        
        # Convert each raw example into one-or-more JSONL lines, then parse
        all_task_examples = []
        for maze_ex in raw_data:
            task_jsonl = dict_to_prompt_completion(maze_ex)
            # Parse lines more efficiently
            for line in task_jsonl.strip().split('\n'):
                if line.strip():
                    all_task_examples.append(json.loads(line))

        # Group by task
        by_task = {}
        for ex in all_task_examples:
            task = ex['task']
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(ex)

        # Max feasible total T given availability and ratios
        caps = []
        for task, ratio in task_ratios.items():
            if ratio <= 0:
                continue
            available = len(by_task.get(task, []))
            # IMPORTANT: include zero-availability tasks (cap becomes 0)
            caps.append(math.floor(available / ratio))
        T = min(len(all_task_examples), min(caps) if caps else 0)

        if T <= 0:
            print(f"Warning: No feasible allocation for {split_name} given ratios and availability.")
            with open(f"{dataset_dir}/{split_name}.jsonl", "w") as f:
                pass
            continue

        # Floors + largest remainders
        counts = {t: int(task_ratios[t] * T) for t in task_ratios}
        leftover = T - sum(counts.values())

        order = sorted(
            task_ratios,
            key=lambda t: (task_ratios[t] * T) - counts[t],
            reverse=True
        )
        for t in order:
            if leftover == 0:
                break
            avail = len(by_task.get(t, []))
            take = min(leftover, max(0, avail - counts[t]))
            if take:
                counts[t] += take
                leftover -= take

        # Defensive clamp (should be no-op if T was computed correctly)
        for t in counts:
            counts[t] = min(counts[t], len(by_task.get(t, [])))

        # Sample, merge, shuffle, save
        chosen = []
        for t, k in counts.items():
            pool = by_task.get(t, [])
            if k > 0 and pool:
                rng.shuffle(pool)
                chosen.extend(pool[:k])
        rng.shuffle(chosen)

        # Write file more efficiently
        output_file = f"{dataset_dir}/{split_name}.jsonl"
        with open(output_file, "w") as f:
            for ex in chosen:
                f.write(json.dumps(ex) + "\n")

        # Count samples per task in final dataset
        task_counts = {}
        for ex in chosen:
            task_counts[ex['task']] = task_counts.get(ex['task'], 0) + 1

        print(f"{split_name}: {len(chosen)} examples (from {len(all_task_examples)}) in {time.time() - start_time:.2f}s")
        print(f"  Task breakdown: {dict(sorted(task_counts.items()))}")

def main(CONFIG=None):
    overall_start = time.time()
    filename = f'{DATA_DIR}/maze_training_{CONFIG["dataset_name"]}.json'
    dataset_dir = f'{DATA_DIR}/{CONFIG["dataset_name"]}'
    if CONFIG["source"] is not None:
        source_filename = f'{DATA_DIR}/maze_training_{CONFIG["source"]}.json'
        source_dataset_dir = f'{DATA_DIR}/{CONFIG["source"]}'

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    splits_data = None
    # Step 1: Check if we should load existing splits
    if CONFIG['skip_full_splits']:
        print("Loading existing splits...")
        splits_data = {}
        if CONFIG["source"] is not None:
            for split_name in CONFIG['splits']:
                splits_data[split_name] = open_jsonl(f'{source_dataset_dir}/{split_name}_full.jsonl')
        else:
            for split_name in CONFIG['splits']:
                splits_data[split_name] = open_jsonl(f'{dataset_dir}/{split_name}_full.jsonl')
    else:
        # Step 2: Generate or load maze examples
        if not CONFIG['skip_generation'] and not os.path.exists(filename):
            print("Generating maze examples...")
            all_examples = generate_maze_examples(CONFIG)
            print(f"Saving {len(all_examples)} examples to {filename}")
            with open(filename, "w") as f:
                json.dump(all_examples, f, indent=2)
        else:
            print("Loading existing maze examples...")
            if CONFIG['source'] is not None:
                with open(source_filename) as f:
                    all_examples = json.load(f)
            else:
                with open(filename) as f:
                    all_examples = json.load(f)
        
        print("Creating train/valid/test splits...")
        split_start = time.time()
        splits_data = data_splitter(all_examples, CONFIG["splits"], CONFIG['split_params'])
        print(f"Splits created in {time.time() - split_start:.2f} seconds")
        
        for split_name, split_data in splits_data.items():
            print(f"Saving {split_name}_full.jsonl with {len(split_data)} examples...")
            save_jsonl(split_data, f'{dataset_dir}/{split_name}_full.jsonl', mapper=dict_to_prompt_completion)

    print("Processing splits with task ratios...")
    process_splits_with_ratios(dataset_dir, CONFIG['task_ratios'], splits_data)

    print("Computing sequence length summary...")
    summary = {
        split: suggest_optimal_max_seq_length(os.path.join(dataset_dir, f"{split}.jsonl"))
        for split in CONFIG['splits']
    }
    summary_path = os.path.join(dataset_dir, "sequence_length_summary.json")
    with open(summary_path, "w") as fout:
        json.dump(summary, fout, indent=2)
    print(f"Sequence length summary written to {summary_path}")
    datasets_len = ""
    for split_name in CONFIG['splits']:
        datasets_len += f"{split_name}: {len(splits_data[split_name])} "
    print(datasets_len)
    print(f"Total time: {time.time() - overall_start:.2f} seconds")

if __name__ == "__main__":
    CONFIG = {
        'random_seed': 123,
        'monte_carlo_params': {
            'repeat': 3,  # Reduced from 5
            'entrances': 5,  # Reduced from 10
            'difficulty': 0.8  # Reduced from 0.9
        },
        'maze_sizes': {
            3: 0,   # Reduced from 10000
            4: 0,  # Reduced from 20000
            5: 50000,  # Reduced from 50000
            6: 0,  # Reduced from 70000
            7: 0,  # Reduced from 100000
        },
        'splits': ['train', 'valid', 'test'],
        'split_params': {
            'test': 0.10,
            'valid': 0.50,
            'random_state': 42
        },
        'task_ratios': {
            "DETECT_START_END": 0,
            "AVAILABLE_DIRECTIONS": 0.2,
            "VALID_MOVE": 0.1,
            "OPTIMAL_NEXT_STEP": 0.7,
            "MAZE_SOLUTION": 0
        },
        'skip_generation': False,
        'skip_full_splits': False,
        'dataset_name': 'curriculum_234', # name_ite_task_mazeSizes
        "source": None 
    }
    print(f"Starting dataset generation with {sum(CONFIG['maze_sizes'].values())} total mazes...")
    main(CONFIG)
