import torch, math, os, json
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

# Updated constants for Mac MPS
MODEL = "Qwen/Qwen3-4B"  # Updated to Qwen3
DATA  = "data/custom_curriculum_1/train.jsonl"
BATCH = 2 # Reduced batch size for MPS memory constraints:
ACCUM = 8

# Learning rate schedule parameters
BASE_LR = 4.0e-5
ITERS = 10000
WARMUP = 0.03 * iters
DECAY_STEPS = iters - warmup
LR_FLOOR = 0.1 * base_lr
EVAL_EVERY = 50

MAXLEN= 512
DEVICE= "mps"  # Force MPS device for Mac

# Checkpoint settings
CHECKPOINT_DIR = "finetuned_model/adapters_dir_qwen3"
RESUME_FROM_CHECKPOINT = False  # Set to False to start fresh

# Check MPS availability
assert torch.backends.mps.is_available(), "MPS not available. Make sure you're on MacOS 12.3+"

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    step_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
    if not step_dirs:
        return None, 0
    
    # Extract step numbers and find the latest
    steps = []
    for step_dir in step_dirs:
        try:
            step_num = int(step_dir.split("_")[1])
            steps.append((step_num, step_dir))
        except (IndexError, ValueError):
            continue
    
    if not steps:
        return None, 0
    
    latest_step, latest_dir = max(steps, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest_dir), latest_step

def save_training_state(checkpoint_dir, step, optimizer, scheduler, loss_history=None):
    """Save complete training state"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history or []
    }
    
    torch.save(state, os.path.join(checkpoint_dir, 'training_state.pt'))

def load_training_state(checkpoint_dir, optimizer, scheduler):
    """Load complete training state"""
    state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    
    if not os.path.exists(state_path):
        print(f"No training state found at {state_path}")
        return 0, []
    
    state = torch.load(state_path, map_location=DEVICE)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    
    step = state.get('step', 0)
    loss_history = state.get('loss_history', [])
    
    print(f"Loaded training state from step {step}")
    return step, loss_history

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

# Check for existing checkpoints
latest_checkpoint, resume_step = find_latest_checkpoint(CHECKPOINT_DIR) if RESUME_FROM_CHECKPOINT else (None, 0)

if latest_checkpoint and RESUME_FROM_CHECKPOINT:
    print(f"Found checkpoint at step {resume_step}: {latest_checkpoint}")
    print("Loading model from checkpoint...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load the fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, latest_checkpoint)
    
    # Ensure LoRA training is enabled (base model frozen, adapters trainable)
    model.train()  # Set to training mode
    for param in model.base_model.parameters():
        param.requires_grad = False  # Freeze base model
    
    # Enable gradients for LoRA adapters only
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    
    print(f"Resumed from checkpoint at step {resume_step}")
    print("LoRA adapters enabled for training, base model frozen")
    
else:
    print("Starting fresh training...")
    
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
    resume_step = 0
    print("Fresh LoRA model created - base model frozen, adapters trainable")

model.to(DEVICE)
model.train()

# Debug: Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                print(f"  Trainable LoRA: {name} - {param.numel()} params")
    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable %: {100 * trainable_params / all_param:.4f}")

print("Model parameter status:")
print_trainable_parameters(model)

# Create custom cosine schedule with minimum LR
def get_cosine_with_min_lr(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
sched = get_cosine_with_min_lr(opt, WARMUP, ITERS, min_lr_ratio=(LR_FLOOR/BASE_LR))

# Load training state if resuming
if latest_checkpoint and RESUME_FROM_CHECKPOINT:
    global_step, loss_history = load_training_state(latest_checkpoint, opt, sched)
else:
    global_step = 0
    loss_history = []

# Training loop - iteration-based instead of epoch-based
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
        
    # Track loss history
    if global_step % 10 == 0:
        loss_history.append((global_step, loss.item() * ACCUM))
    
    # Save checkpoint every EVAL_EVERY steps
    if global_step > 0 and global_step % EVAL_EVERY == 0:
        checkpoint_dir = f"{CHECKPOINT_DIR}/step_{global_step}"
        model.save_pretrained(checkpoint_dir)
        save_training_state(checkpoint_dir, global_step, opt, sched, loss_history)
        print(f"Saved checkpoint at step {global_step}")
        
    global_step += 1

# Final save
print(f"Training completed after {ITERS} iterations!")
final_dir = f"{CHECKPOINT_DIR}/final"
model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
save_training_state(final_dir, global_step, opt, sched, loss_history)
print(f"Final model saved to {final_dir}")

# Print loss history summary
if loss_history:
    print("\nLoss History Summary:")
    for step, loss_val in loss_history[-10:]:  # Show last 10 entries
        print(f"  Step {step}: Loss {loss_val:.6f}")
