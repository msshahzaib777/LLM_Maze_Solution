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
    min_loss_step = min(range(len(loss_history)), key=lambda i: loss_history[i])
    best_step = min_loss_step + 1  # Adding 1 because steps typically start from 1
    best_checkpoint = os.path.join(adapter_base_path, f"checkpoint-{best_step}")
    print(f"Loading best checkpoint: checkpoint-{best_step} with loss: {loss_history[min_loss_step]}")

# Load the model with best checkpoint
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto")
model = PeftModel.from_pretrained(base, best_checkpoint)
model.eval()

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
BATCH_SIZE = 8

# Generate responses in batches
print("Generating responses...\n")
for i in tqdm(range(0, len(test_examples), BATCH_SIZE), desc="Generating", unit="batch"):
    batch = test_examples[i:i + BATCH_SIZE]
    prompts = [example['prompt'] for example in batch]
    
    # Tokenize batch
    inputs = tok(prompts, padding=True, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_config
        )
    
    # Decode and print responses
    responses = tok.batch_decode(outputs, skip_special_tokens=True)
    
    for j, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"Example {i+j+1}:")
        print(f"Prompt: {prompt}\n")
        print(f"Generated Response: {response}\n")
        print("-" * 80 + "\n")
