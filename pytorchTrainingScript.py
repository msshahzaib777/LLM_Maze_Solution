import torch, math
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_with_min_lr_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Updated constants for Mac MPS
MODEL = "Qwen/Qwen3-4B"  # Updated to Qwen3
DATA  = "data/custom_curriculum_1/train.jsonl"
BATCH = 16  # Reduced batch size for MPS memory constraints
ACCUM = 4

# Learning rate schedule parameters
BASE_LR = 4.0e-5
ITERS = 1000
WARMUP = 300  # 0.03 * iters
DECAY_STEPS = 9700  # iters - warmup
LR_FLOOR = 4.0e-6  # 0.1 * base_lr
EVAL_EVERY = 50

MAXLEN= 512
DEVICE= "mps"  # Force MPS device for Mac

# Check MPS availability
assert torch.backends.mps.is_available(), "MPS not available. Make sure you're on MacOS 12.3+"

ds = load_dataset("json", data_files={"train": DATA})

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def collate_fn(batch):
    """Process and tokenize data on-the-go"""
    # Format examples: combine prompt + completion
    texts = [ex["prompt"] + ex["completion"] for ex in batch]
    
    # Tokenize on the fly
    tokenized = tok(texts, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

dl = DataLoader(ds["train"], batch_size=BATCH, shuffle=True, collate_fn=collate_fn)

# Load model with float16 for MPS
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    dtype=torch.float16,
    trust_remote_code=True
)

# Configure LoRA
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.02,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)
model.to(DEVICE)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
sched = get_cosine_with_min_lr_schedule_with_warmup(opt, WARMUP, ITERS, min_lr=BASE_LR)

# Training loop - iteration-based instead of epoch-based
global_step = 0
opt.zero_grad()

# Create infinite iterator from dataloader
def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

data_iter = infinite_dataloader(dl)

print(f"Starting training for {ITERS} iterations...")
print(f"Base LR: {BASE_LR}, Warmup: {WARMUP}, LR Floor: {LR_FLOOR}")

while global_step < ITERS:
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(DEVICE)
    attn = batch["attention_mask"].to(DEVICE)
    
    out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
    loss = out.loss / ACCUM
    loss.backward()
    
    if (global_step + 1) % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        
    if global_step % 50 == 0:  # Print loss every 50 steps
        current_lr = sched.get_last_lr()[0]
        print(f"Step: {global_step}/{ITERS}, Loss: {loss.item()*ACCUM:.6f}, LR: {current_lr:.2e}")
        
    # Save checkpoint every EVAL_EVERY steps
    if global_step > 0 and global_step % EVAL_EVERY == 0:
        checkpoint_dir = f"finetuned_model/adapters_dir_qwen3/step_{global_step}"
        model.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint at step {global_step}")
        
    global_step += 1

# Final save
print(f"Training completed after {ITERS} iterations!")
final_dir = "finetuned_model/adapter/adapter_dir_qwen3/final"
model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
print(f"Final model saved to {final_dir}")
