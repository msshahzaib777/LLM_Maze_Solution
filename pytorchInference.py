import torch, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, InfNanRemoveLogitsProcessor
from peft import PeftModel
from tqdm import tqdm

# ----- Device & dtype -----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    target_dtype = torch.bfloat16   # switch to float32 if you still see instability
    print("Using MPS backend (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    target_dtype = torch.float16
    print("Using CUDA backend")
else:
    device = torch.device("cpu")
    target_dtype = torch.float32
    print("Using CPU backend")

base_id = "Qwen/Qwen3-4B"
adapter_base_path = "./finetuned_model/adapters_dir_qwen3"


# ----- pick best checkpoint (your logic preserved) -----
checkpoint_dirs = [d for d in os.listdir(adapter_base_path) if os.path.isdir(os.path.join(adapter_base_path, d))]
last_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))[-1]
last_checkpoint_path = os.path.join(adapter_base_path, last_checkpoint)
training_state = torch.load(os.path.join(last_checkpoint_path, 'training_state.pt'))
loss_history = training_state.get('loss_history', [])
if not loss_history:
    print("No loss history found, using last checkpoint")
    best_checkpoint = last_checkpoint_path
else:
    min_loss_step, min_loss = min(loss_history, key=lambda x: x[1])
    steps = [int(d.split('_')[1]) for d in checkpoint_dirs]
    nearest_step = min(steps, key=lambda x: abs(x - min_loss_step))
    best_checkpoint = os.path.join(adapter_base_path, f"{checkpoint_dirs[0].split('_')[0]}_{nearest_step}")
    print(f"Loading best checkpoint: checkpoint-{nearest_step} with loss: {min_loss} (from step {min_loss_step})")

# ----- tokenizer -----
tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ----- load base on CPU to merge safely, then move -----
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32)  # merge in fp32
peft_model = PeftModel.from_pretrained(base, best_checkpoint, dtype=torch.float32, is_trainable=False)

# ----- MERGE -----
merged = peft_model.merge_and_unload()   # LoRA weights baked into base; no PEFT wrappers left

# (Optional) cast to your runtime dtype and move to device
merged = merged.to(device, dtype=target_dtype)
merged.eval()

# ----- generation knobs -----
logits_processor = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
gen_config = dict(
    max_new_tokens=127,
    do_sample=True,
    temperature=0.7,   # > 0
    top_p=0.9,
    num_beams=1,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)

# Setup evaluation directory
eval_dir = os.path.join(f"results/{'/'.join(best_checkpoint.split('/')[-2:])}", "eval_1")
os.makedirs(eval_dir, exist_ok=True)
preds_jsonl = os.path.join(eval_dir, "test_predictions.jsonl")
summary_json = os.path.join(eval_dir, "summary.json")

# Load and process test examples
test_file = "data/custom_curriculum_1/test.jsonl"
with open(test_file, 'r') as f:
    test_examples = [json.loads(line) for line in f]

# Generation parameters - conservative settings for MPS stability
gen_config = {
    "max_new_tokens": 127,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tok.pad_token_id,
    "num_beams": 1
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
            
            inputs = tok(prompts, padding=True, return_tensors="pt").to(device)  # keep ids/mask as integers
            
            with torch.no_grad():
                outputs = merged.generate(
                    **inputs,
                    **gen_config,
                    logits_processor=logits_processor
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

# ----- example batch (optional test) -----
# prompts = ["Solve this 5x5 maze..."]
# inputs = tok(prompts, padding=True, return_tensors="pt").to(device)  # keep ids/mask as integers
# 
# with torch.no_grad():
#     out = merged.generate(**inputs, **gen_config, logits_processor=logits_processor)
# 
# print(tok.batch_decode(out, skip_special_tokens=True))