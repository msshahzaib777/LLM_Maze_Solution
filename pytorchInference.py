import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json, os
from tqdm import tqdm

base_id = "Qwen/Qwen3-4B"          # example
adapter_base_path = "./finetuned_model/adapters_dir_qwen3"     # PEFT-style adapter

# Get the last checkpoint directory
checkpoint_dirs = [d for d in os.listdir(adapter_base_path) if os.path.isdir(os.path.join(adapter_base_path, d))]
last_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))[-1]  # Sort by step number
last_checkpoint_path = os.path.join(adapter_base_path, last_checkpoint)

# Load the training state from the last checkpoint
training_state = torch.load(os.path.join(last_checkpoint_path, 'training_state.pt'))
loss_history = training_state.get('loss_history', [])

if not loss_history:
    print("No loss history found, using last checkpoint")
    best_checkpoint = last_checkpoint_path
else:
    # Find the checkpoint with minimum loss
    min_loss_step, min_loss = min(loss_history, key=lambda x: x[1])
    # Get all checkpoint steps
    checkpoint_steps = [int(d.split('_')[1]) for d in checkpoint_dirs]
    # Find the nearest available checkpoint to min_loss_step
    nearest_checkpoint_step = min(checkpoint_steps, key=lambda x: abs(x - min_loss_step))
    best_checkpoint = os.path.join(adapter_base_path, f"{checkpoint_dirs[0].split('_')[0]}_{nearest_checkpoint_step}")
    print(f"Loading best checkpoint: checkpoint-{nearest_checkpoint_step} with loss: {min_loss} (from step {min_loss_step})")

# Load the model with best checkpoint
tok = AutoTokenizer.from_pretrained(base_id)
tok.padding_side = 'left'  # Set left padding for decoder-only models
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto")
model = PeftModel.from_pretrained(base, best_checkpoint)
model.eval()

# Setup evaluation directory
eval_dir = os.path.join(best_checkpoint, "eval_1")
os.makedirs(eval_dir, exist_ok=True)
preds_jsonl = os.path.join(eval_dir, "test_predictions.jsonl")
summary_json = os.path.join(eval_dir, "summary.json")

# Load and process test examples
test_file = "data/custom_curriculum_1/test.jsonl"
with open(test_file, 'r') as f:
    test_examples = [json.loads(line) for line in f]

# Generation parameters
gen_config = {
    "max_new_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tok.pad_token_id
}

# Batch size for processing
BATCH_SIZE = 6

# Load existing predictions if file exists
existing_predictions = set()
if os.path.exists(preds_jsonl):
    with open(preds_jsonl, 'r') as f:
        for line in f:
            pred = json.loads(line)
            existing_predictions.add(pred['id'])
    print(f"Found {len(existing_predictions)} existing predictions")

# Filter out examples that already have predictions
test_examples = [ex for ex in test_examples if ex['id'] not in existing_predictions]
if not test_examples:
    print("All examples have already been processed")
    exit(0)

# Group examples by maze size and task
print("Grouping examples by maze size and task...\n")
grouped_examples = {}
for example in test_examples:
    maze_size = example['id'].split('_')[0]
    task = example.get('task', 'UNKNOWN')
    key = (maze_size, task)
    if key not in grouped_examples:
        grouped_examples[key] = []
    grouped_examples[key].append(example)

# Generate responses in batches for each group
print("Generating responses...\n")
total_examples = len(test_examples)
processed = 0

with open(preds_jsonl, 'a') as outfile:  # Open in append mode
    pbar = tqdm(total=total_examples, 
                desc="Evaluating", 
                unit="example",
                postfix={"examples": 0})
    
    for (maze_size, task), group in grouped_examples.items():
        print(f"\nProcessing {maze_size}, {task} - {len(group)} examples")
        
        for i in range(0, len(group), BATCH_SIZE):
            batch = group[i:i + BATCH_SIZE]
            prompts = [example['prompt'] for example in batch]
            
            inputs = tok(prompts, padding=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_config
                )
            
            responses = tok.batch_decode(outputs, skip_special_tokens=True)
            
            generated_responses = []
            for prompt, response in zip(prompts, responses):
                if response.startswith(prompt):
                    generated_part = response[len(prompt):].strip()
                else:
                    generated_part = response.strip()
                generated_responses.append(generated_part)
            
            for j, (item, prompt, generated_response) in enumerate(zip(batch, prompts, generated_responses)):
                pred_item = {
                    'id': item.get('id', processed + j),
                    'prompt': item['prompt'],
                    'target': item.get('completion', ''),
                    'prediction': generated_response,
                    'task': item.get('task', 'UNKNOWN')
                }
                outfile.write(json.dumps(pred_item) + '\n')
            
            processed += len(batch)
            pbar.update(len(batch))
            pbar.set_postfix({"examples": processed})

print(f"\nPredictions saved to: {preds_jsonl}")
print(f"Total new examples processed: {processed}")