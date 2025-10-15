import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json, os

base_id = "Qwen/Qwen3-4B"          # example
adapter_base_path = "./finetuned_model/adapters_dir_qwen3"     # PEFT-style adapter

# Find all checkpoint directories
checkpoint_dirs = [d for d in os.listdir(adapter_base_path) if os.path.isdir(os.path.join(adapter_base_path, d))]

# Track best checkpoint
best_loss = float('inf')
best_checkpoint = None
best_step = None

# Iterate through checkpoints
for checkpoint in checkpoint_dirs:
    checkpoint_path = os.path.join(adapter_base_path, checkpoint)
    training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
    
    if os.path.exists(training_state_path):
        # Load training state
        training_state = torch.load(training_state_path)
        current_loss = training_state.get('loss', float('inf'))
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_checkpoint = checkpoint_path
            best_step = checkpoint

print(f"Loading best checkpoint: {best_step} with loss: {best_loss}")

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
for i in range(0, len(test_examples), BATCH_SIZE):
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
