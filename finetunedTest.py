# --------------------------
# Batched eval on test.jsonl + save per-example outputs
# --------------------------
import os, json, sys
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlx.core as mx
from tqdm.auto import tqdm

from mlx_lm import load, generate, batch_generate

# --------------------------
# Paths & constants
# --------------------------
ds_dir = "data/custom_curriculum_1"
adapter_dir = "finetuned_model/adapter/adapters_merged_2"
model_path = "finetuned_model/models/Qwen3-4B-MLX-bf16_start_end"

ALLOWED_DIRS = {"up", "down", "left", "right"}
test_path = os.path.join(ds_dir, "test.jsonl")

# Tune batch throughput vs. memory
# Larger batch size for systems with high RAM (64GB)
BATCH_SIZE = 4  # Increased batch size for faster processing with sufficient memory
MAX_TOKENS = 512          # decoding budget per sample
TEMPERATURE = 0.0        # deterministic
TOP_P = 1.0
SEED = 0
VERBOSE = False

# --------------------------
# Load model
# --------------------------
model_lora, tokenizer = load(
    model_path,
    tokenizer_config={"trust_remote_code": True},
    adapter_path=adapter_dir
)

model_lora.eval()
mx.random.seed(SEED)

# --------------------------
# Load test data & run inference
# --------------------------
eval_dir = os.path.join(adapter_dir, "eval_1")
os.makedirs(eval_dir, exist_ok=True)
preds_jsonl = os.path.join(eval_dir, "test_predictions.jsonl")
summary_json = os.path.join(eval_dir, "summary.json")

def batch_inference():
    # Initialize results dictionary
    results = {
        'DETECT_START_END': {'correct': 0, 'total': 0},
        'AVAILABLE_DIRECTIONS': {'correct': 0, 'total': 0}, 
        'VALID_MOVE': {'correct': 0, 'total': 0},
        'OPTIMAL_NEXT_STEP': {'correct': 0, 'total': 0}
    }

    # Load test data
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
        test_data = [item for item in test_data if item['task'] == 'OPTIMAL_NEXT_STEP']
    # Check for existing predictions and resume if possible
    existing_preds = []
    if os.path.exists(preds_jsonl):
        with open(preds_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    existing_preds.append(json.loads(line))
        completed = len(existing_preds)
        if completed >= len(test_data):
            print(f"Predictions already complete at {preds_jsonl}, skipping inference...")
            return existing_preds
        else:
            print(f"Resuming from {completed} / {len(test_data)} predictions in {preds_jsonl}...")
            
            # Update results with existing predictions by matching with test data
            for pred, test_item in zip(existing_preds, test_data):
                task = test_item['task']
                if task == 'DETECT_START_END':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_start_end_task(pred)
                elif task == 'AVAILABLE_DIRECTIONS':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_directions_task(pred)
                elif task == 'VALID_MOVE':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_valid_move_task(pred)
                elif task == 'OPTIMAL_NEXT_STEP':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_optimal_step_task(pred)
    else:
        completed = 0

    predictions = existing_preds.copy()
    total_batches = (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE
    with open(preds_jsonl, 'a') as outfile:
        for batch_start in tqdm(
            range(completed, len(test_data), BATCH_SIZE),
            total=total_batches - (completed // BATCH_SIZE),
            desc="Evaluating",
            unit="batch"
        ):
            batch = test_data[batch_start:batch_start + BATCH_SIZE]
            prompts = [item['prompt'] for item in batch]
            tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]

            outputs = batch_generate(
                model_lora,
                tokenizer,
                prompts=tokenized_prompts,
                max_tokens=MAX_TOKENS,
                verbose=VERBOSE,
            )

            for offset, (item, output) in enumerate(zip(batch, outputs.texts)):
                pred_item = {
                    'id': item.get('id', batch_start + offset),
                    'prompt': item['prompt'],
                    'target': item.get('completion', ''),
                    'prediction': output,
                    'task': item.get('task', 'UNKNOWN')  
                }
                predictions.append(pred_item)
                outfile.write(json.dumps(pred_item) + '\n')
                
                # Evaluate after each prediction
                task = pred_item['task']
                if task == 'DETECT_START_END':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_start_end_task(pred_item)
                elif task == 'AVAILABLE_DIRECTIONS':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_directions_task(pred_item)
                elif task == 'VALID_MOVE':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_valid_move_task(pred_item)
                elif task == 'OPTIMAL_NEXT_STEP':
                    results[task]['total'] += 1
                    results[task]['correct'] += evaluate_optimal_step_task(pred_item)

            # Print current results after each batch
            print("\nCurrent Results:")
            for task, counts in results.items():
                if counts['total'] > 0:
                    accuracy = counts['correct'] / counts['total']
                    print(f"{task}: {accuracy:.2%} ({counts['correct']}/{counts['total']})")

    # Save final results
    summary = {}
    for task, counts in results.items():
        if counts['total'] > 0:
            accuracy = counts['correct'] / counts['total']
            summary[task] = {
                'accuracy': accuracy,
                'correct': counts['correct'],
                'total': counts['total']
            }

    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nFinal Evaluation Results:")
    for task, metrics in summary.items():
        print(f"{task}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

    return predictions

# Helper functions for evaluation
def extract_json_after_think(text):
    try:
        # Find the position after </think>
        think_end = text.find('</think>')
        if think_end != -1:
            json_text = text[think_end + 8:].strip()  # +8 for '</think>'
            return json.loads(json_text)
        return json.loads(text)  # fallback to parsing whole text
    except:
        return {}

def evaluate_start_end_task(pred_item):
    try:
        target = extract_json_after_think(pred_item['target'])
        prediction = extract_json_after_think(pred_item['prediction'])
        return (target['start'] == prediction.get('start') and 
                target['end'] == prediction.get('end'))
    except:
        return False

def evaluate_directions_task(pred_item):
    try:
        target = extract_json_after_think(pred_item['target'])
        prediction = extract_json_after_think(pred_item['prediction'])
        target_dirs = set(target['available_directions'])
        pred_dirs = set(prediction.get('available_directions', []))
        return target_dirs == pred_dirs
    except:
        return False

def evaluate_valid_move_task(pred_item):
    try:
        target = pred_item['target']
        prediction = extract_json_after_think(pred_item['prediction'])
        return target['is_valid'] == prediction.get('is_valid')
    except:
        return False

def evaluate_optimal_step_task(pred_item):
    try:
        target = extract_json_after_think(pred_item['target'])
        prediction = extract_json_after_think(pred_item['prediction'])
        return target['optimal_step'] == prediction.get('optimal_step')
    except:
        return False

# Run inference and evaluation
predictions = batch_inference()