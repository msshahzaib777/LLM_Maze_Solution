import json
from sklearn.model_selection import train_test_split
from utils import (
    dict_to_prompt_completion,
    make_training_example,
    TASKS,
    filter_jsonl_by_task_ratio,
    suggest_optimal_max_seq_length,
    open_jsonl,
    save_jsonl
)
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import os
import argparse

def generate_maze_examples(config):
    """Generate maze examples based on size counts"""
    Maze.set_seed(123)
    all_examples = []
    for g, n in config['maze_sizes'].items():
        for count in range(n):
            m = Maze()
            m.generator = Prims(g, g)
            m.solver = BacktrackingSolver()
            if count < n//4:
                m.generate_monte_carlo(config["monte_carlo_params"]["repeat"], config["monte_carlo_params"]["entrances"], config["monte_carlo_params"]['difficulty'])
            else:
                m.generate()
                m.generate_entrances(start_outer=False, end_outer=False)
            m.solve()
            all_examples.append(make_training_example(m, TASKS, id=f"{g}x{g}_{count}"))
    return all_examples

def split_data(data):
    """Split data into train, valid, and test sets"""
    labels = [d["maze_size"] for d in data]
    indices = list(range(len(data)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=42
    )

    temp_data = [data[i] for i in temp_idx]
    temp_labels = [d["maze_size"] for d in temp_data]
    temp_indices = list(range(len(temp_data)))
    valid_idx, test_idx = train_test_split(
        temp_indices, test_size=0.33, stratify=temp_labels, random_state=42
    )

    train_data = [data[i] for i in train_idx]
    valid_data = [temp_data[i] for i in valid_idx]
    test_data = [temp_data[i] for i in test_idx]
    
    return train_data, valid_data, test_data

def process_splits_with_ratios(dataset_dir, task_ratios, splits):
    """Process splits with given task ratios"""
    for split_name, split_data in splits.items():
        filtered_jsonl = filter_jsonl_by_task_ratio(
            input_data=split_data,
            task_ratios=task_ratios
        )
        
        output_path = f'{dataset_dir}/{split_name}.jsonl'
        with open(output_path, "w") as f:
            f.write(filtered_jsonl)

def main(CONFIG=None):
    filename = f'./data/maze_training_{CONFIG["dataset_name"]}.json'
    dataset_dir = f'./data/{CONFIG["dataset_name"]}'

    os.makedirs("./data", exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # Step 1: Generate or load maze examples
    if not CONFIG['skip_generation'] and not os.path.exists(filename):
        print("Generating maze examples...")
        all_examples = generate_maze_examples(CONFIG['maze_sizes'])
        with open(filename, "w") as f:
            json.dump(all_examples, f, indent=2)
    else:
        print("Loading existing maze examples...")
        with open(filename) as f:
            all_examples = json.load(f)
        
    train_data = valid_data = test_data = None
    # Step 2: Create or load full splits
    if not CONFIG['skip_full_splits']:
        print("Creating train/valid/test splits...")
        train_data, valid_data, test_data = split_data(all_examples)
        save_jsonl(train_data, f'{dataset_dir}/train_full.jsonl', mapper=dict_to_prompt_completion)
        save_jsonl(valid_data, f'{dataset_dir}/valid_full.jsonl', mapper=dict_to_prompt_completion)
        save_jsonl(test_data, f'{dataset_dir}/test_full.jsonl', mapper=dict_to_prompt_completion)
    else:
        print("Loading existing splits...")
        train_data = open_jsonl(f'{dataset_dir}/train_full.jsonl')
        valid_data = open_jsonl(f'{dataset_dir}/valid_full.jsonl')
        test_data = open_jsonl(f'{dataset_dir}/test_full.jsonl')

    print("Processing splits with task ratios...")
    process_splits_with_ratios(dataset_dir, CONFIG['task_ratios'], CONFIG['splits'])

    summary = {
        split: suggest_optimal_max_seq_length(os.path.join(dataset_dir, f"{split}.jsonl"))
        for split in CONFIG['splits']
    }
    summary_path = os.path.join(dataset_dir, "sequence_length_summary.json")
    with open(summary_path, "w") as fout:
        json.dump(summary, fout, indent=2)
    print(f"Sequence length summary written to {summary_path}")
    
    print(f"Wrote {len(train_data)} train and {len(valid_data)} valid and {len(test_data)} test examples")

if __name__ == "__main__":
    CONFIG = {
        'random_seed': 123,
        'monte_carlo_params': {
            'repeat': 5,
            'entrance': 10,
            'difficulty': 0.5
        },
        'maze_sizes': {
            3: 500,
            4: 5000,
            5: 7000,
            6: 10000,
            7: 15000,
        },
        'splits': ['train', 'valid', 'test'],
        'split_params': {
            'test_size': 0.30,
            'valid_test_split': 0.33,
            'random_state': 42
        },
        'task_ratios': {
            "DETECT_START_END": 0.1,
            "AVAILABLE_DIRECTIONS": 0.2,
            "VALID_MOVE": 0.2,
            "OPTIMAL_NEXT_STEP": 0.5
        },
        'skip_generation': False,
        'skip_full_splits': False,
        'dataset_name': 'curriculum_1'
    }
    main(CONFIG)
