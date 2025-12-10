#!/usr/bin/env python3
"""
Minimal script to calculate accuracy from test_predictions.jsonl
Supports different tasks and maze sizes with detailed breakdowns.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

def extract_json_from_response(text):
    """Extract JSON from response, handling <think> blocks."""
    if not text:
        return None
    
    # Remove <think> blocks
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text_clean = text_clean.strip()
    
    # Try to find JSON in the remaining text
    # Look for {...} patterns
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_clean)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If no JSON found, try to parse the entire cleaned text
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        return None

def extract_maze_size(id_str):
    """Extract maze size from ID like '3x3_7575_valid_move_51' -> '3x3'"""
    match = re.match(r'(\d+x\d+)', id_str)
    return match.group(1) if match else 'unknown'

def compare_responses(target, prediction, task="UNKNOWN", debug=False):
    """Compare target and prediction responses."""
    target_json = extract_json_from_response(target)
    pred_json = extract_json_from_response(prediction)
    
    if target_json is None or pred_json is None:
        if debug:
            print(f"DEBUG: Failed to extract JSON - Target: {target_json}, Pred: {pred_json}")
        return False, 0.0
    
    # Compare the JSON objects (excluding 'think' field for comparison)
    target_clean = {k: v for k, v in target_json.items() if k != 'think'}
    pred_clean = {k: v for k, v in pred_json.items() if k != 'think'}
    
    if debug:
        print(f"DEBUG: Target: {target_clean}")
        print(f"DEBUG: Prediction: {pred_clean}")
    
    exact_match = target_clean == pred_clean
    
    # Calculate partial path correctness only for MAZE_SOLUTION task
    path_similarity = 0.0
    if task == "MAZE_SOLUTION" and 'path' in target_clean and 'path' in pred_clean:
        target_path = target_clean['path']
        pred_path = pred_clean['path']
        if target_path and pred_path:
            min_len = min(len(target_path), len(pred_path))
            correct_steps = sum(1 for i in range(min_len) if target_path[i] == pred_path[i])
            path_similarity = correct_steps / len(target_path) if target_path else 0.0
    
    return exact_match, path_similarity

def calculate_accuracy(jsonl_file, verbose=True, save_text_report=False):
    """Calculate accuracy metrics from predictions file."""
    
    # Initialize counters
    total_examples = 0
    correct_examples = 0
    total_path_similarity = 0.0
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'path_sim': 0.0})
    size_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'path_sim': 0.0})
    task_size_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0, 'path_sim': 0.0}))
    
    # Store text output for saving to file
    text_output = []
    
    if verbose:
        print(f"Processing predictions from: {jsonl_file}")
        print("-" * 60)
    
    if save_text_report:
        text_output.append(f"Processing predictions from: {jsonl_file}")
        text_output.append("-" * 60)
    
    # Process each line
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Extract information
                example_id = data.get('id', f'line_{line_num}')
                task = data.get('task', 'UNKNOWN')
                target = data.get('target', '')
                prediction = data.get('prediction', '')
                
                # Extract maze size
                maze_size = extract_maze_size(example_id)
                
                # Check if prediction is correct (debug first few examples)
                debug_mode = line_num <= 3
                is_correct, path_sim = compare_responses(target, prediction, task, debug_mode)
                
                # Update counters
                total_examples += 1
                total_path_similarity += path_sim
                if is_correct:
                    correct_examples += 1
                
                # Update task statistics
                task_stats[task]['total'] += 1
                task_stats[task]['path_sim'] += path_sim
                if is_correct:
                    task_stats[task]['correct'] += 1
                
                # Update size statistics
                size_stats[maze_size]['total'] += 1
                size_stats[maze_size]['path_sim'] += path_sim
                if is_correct:
                    size_stats[maze_size]['correct'] += 1
                
                # Update task-size statistics
                task_size_stats[task][maze_size]['total'] += 1
                task_size_stats[task][maze_size]['path_sim'] += path_sim
                if is_correct:
                    task_size_stats[task][maze_size]['correct'] += 1
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                if verbose:
                    print(f"Error processing line {line_num}: {e}")
                continue
    
    # Calculate and display results
    overall_acc = (correct_examples / total_examples * 100) if total_examples > 0 else 0
    avg_path_sim = (total_path_similarity / total_examples * 100) if total_examples > 0 else 0
    
    # Prepare text output
    overall_text = [
        f"\n=== OVERALL ACCURACY ===",
        f"Total Examples: {total_examples}",
        f"Correct: {correct_examples}",
        f"Overall Accuracy: {overall_acc:.2f}%",
        f"Average Path Similarity: {avg_path_sim:.2f}%"
    ]
    
    task_text = [f"\n=== ACCURACY BY TASK ==="]
    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        path_acc = (stats['path_sim'] / stats['total'] * 100) if stats['total'] > 0 else 0
        task_text.append(f"{task:20s}: {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%) Path: {path_acc:5.1f}%")
    
    size_text = [f"\n=== ACCURACY BY MAZE SIZE ==="]
    for size in sorted(size_stats.keys()):
        stats = size_stats[size]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        size_text.append(f"{size:10s}: {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%)")
    
    task_size_text = [f"\n=== ACCURACY BY TASK AND MAZE SIZE ==="]
    for task in sorted(task_size_stats.keys()):
        task_size_text.append(f"\n{task}:")
        for size in sorted(task_size_stats[task].keys()):
            stats = task_size_stats[task][size]
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            task_size_text.append(f"  {size:8s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")
    
    # Print if verbose
    if verbose:
        for line in overall_text:
            print(line)
        for line in task_text:
            print(line)
        for line in size_text:
            print(line)
        for line in task_size_text:
            print(line)
    
    # Add to text output if saving
    if save_text_report:
        text_output.extend(overall_text)
        text_output.extend(task_text)
        text_output.extend(size_text)
        text_output.extend(task_size_text)
    
    # Prepare results for JSON serialization
    results = {
        'overall': {'correct': correct_examples, 'total': total_examples, 'accuracy': overall_acc},
        'by_task': {},
        'by_size': {},
        'by_task_size': {}
    }
    
    # Convert task stats
    for task, stats in task_stats.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        results['by_task'][task] = {
            'correct': stats['correct'],
            'total': stats['total'],
            'accuracy': acc
        }
    
    # Convert size stats
    for size, stats in size_stats.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        results['by_size'][size] = {
            'correct': stats['correct'],
            'total': stats['total'],
            'accuracy': acc
        }
    
    # Convert task-size stats
    for task, size_dict in task_size_stats.items():
        results['by_task_size'][task] = {}
        for size, stats in size_dict.items():
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            results['by_task_size'][task][size] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': acc
            }
    
    return results, text_output if save_text_report else None

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python calculate_accuracy.py <path_to_test_predictions.jsonl>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    
    if not Path(jsonl_file).exists():
        print(f"Error: File {jsonl_file} not found!")
        sys.exit(1)
    
    try:
        results, text_output = calculate_accuracy(jsonl_file, verbose=True, save_text_report=True)
        print(f"\n=== SUMMARY ===")
        print(f"Overall Accuracy: {results['overall']['accuracy']:.2f}%")
        
        # Save results to JSON file in same directory
        jsonl_path = Path(jsonl_file)
        results_file = jsonl_path.parent / "accuracy_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save text report to file
        if text_output:
            text_report_file = jsonl_path.parent / "accuracy_report.txt"
            with open(text_report_file, 'w') as f:
                f.write('\n'.join(text_output))
            print(f"\nResults saved to: {results_file}")
            print(f"Text report saved to: {text_report_file}")
        else:
            print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()