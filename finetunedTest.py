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
BATCH_SIZE = 64  # Increased batch size for faster processing with sufficient memory
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
    # Check if predictions already exist
    # Load test data
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

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
                    'prediction': output
                }
                predictions.append(pred_item)
                outfile.write(json.dumps(pred_item) + '\n')

    return predictions

# Run inference
predictions = batch_inference()

# --------------------------
# Evaluate predictions by task type
# --------------------------
def evaluate_start_end_task(pred_item):
    # Extract ground truth and prediction
    try:
        target = json.loads(pred_item['target'])
        prediction = json.loads(pred_item['prediction'])
        return (target['start'] == prediction['start'] and 
                target['end'] == prediction['end'])
    except:
        return False

def evaluate_directions_task(pred_item):
    try:
        target = json.loads(pred_item['target'])
        prediction = json.loads(pred_item['prediction'])
        return target['available_directions'] == prediction['available_directions']
    except:
        return False

def evaluate_valid_move_task(pred_item):
    try:
        target = json.loads(pred_item['target'])
        prediction = json.loads(pred_item['prediction'])
        return target['is_valid'] == prediction['is_valid']
    except:
        return False

def evaluate_optimal_step_task(pred_item):
    try:
        target = json.loads(pred_item['target'])
        prediction = json.loads(pred_item['prediction'])
        return target['optimal_step'] == prediction['optimal_step']
    except:
        return False

# Calculate metrics per task
results = {
    'DETECT_START_END': {'correct': 0, 'total': 0},
    'AVAILABLE_DIRECTIONS': {'correct': 0, 'total': 0}, 
    'VALID_MOVE': {'correct': 0, 'total': 0},
    'OPTIMAL_NEXT_STEP': {'correct': 0, 'total': 0}
}

for pred in predictions:
    task = json.loads(pred['prompt'])['task']
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

# Calculate accuracies and save results
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

print("\nEvaluation Results:")
for task, metrics in summary.items():
    print(f"{task}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
