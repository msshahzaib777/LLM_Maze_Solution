#!/usr/bin/env python3
# Run with: torchrun --nproc_per_node=2 ddpTraining.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import logging
from constant import ADAPTER_DIR, DATA_DIR, MERGED_MODEL_DIR, ROOT_DIR
from plot_learning_curves import plot_losses
from datetime import datetime
import constant as config
from transformers import get_cosine_with_min_lr_schedule_with_warmup

# Setup - Use free GPUs (0 and 3) with NCCL fixes
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU 1,2 are occupied
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 min timeout
os.environ["NCCL_DEBUG"] = "INFO"  # Debug NCCL issues
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand if causing issues
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P if causing issues

dist.init_process_group("nccl", timeout=torch.distributed.default_pg_timeout)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Config - Memory optimized

MODEL = config.base_id
# MODEL = MERGED_MODEL_DIR + "/qwen3_1_123_merged"
DATASET_NAME = config.Dataset_NAME
adapter_name = config.ADAPTER_NAME
INCLUDE_REASONING = True  # Flag to include <think> tags in completion
DATA = DATA_DIR + f"/{DATASET_NAME}/train.jsonl"
EPOCH = config.EPOCH  # Increased steps for better convergence
LOG_STEP = 50  # Less frequent logging to reduce overhead
EVAL_STEP = config.EVAL_STEPS # Less frequent evaluation for faster training
SAVE_STEP = 200  # Less frequent saving to reduce I/O overhead
BASE_LR = 5.0e-6  # More conservative learning rate for stability
ACCUM = config.ACCUM  # Higher accumulation for better gradient estimates
BATCH_SIZE = config.BATCH_SIZE  # Smaller batch size for larger sequences
MIN_LR = config.MIN_LR  # Minimum learning rate for scheduler
# Calculate steps based on epoch, batch size, accum, and training samples count
num_samples = 757668
steps_per_epoch = (num_samples + BATCH_SIZE * ACCUM - 1) // (BATCH_SIZE * ACCUM)
STEPS = EPOCH * steps_per_epoch
WARMUP = int(0.05 * STEPS)  # Longer warmup period (5% of steps)
MAX_SEQ_LEN = 256
CHECKPOINT_DIR = f"{ADAPTER_DIR}/{adapter_name}"
RESUME = False
resume_step = config.step_num  # Set to 0 to auto-detect latest checkpoint

# Early stopping config
EARLY_STOP_PATIENCE = 100  # Stop if no improvement for 5 eval steps
best_val_loss = float('inf')
no_improve_count = 0

# 8-bit quantization for memory efficiency
USE_8BIT = config.quantized
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
) if USE_8BIT else None
# Verify GPU setup
if local_rank == 0:
    print(f"Physical GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current device: {torch.cuda.current_device()}")

logger = None
if local_rank == 0:
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created at: {log_file}")

# Data - token-based batching
ds = load_dataset("json", data_files={"train": DATA})
if local_rank == 0:
    logger.info("Sample training data:")
    logger.info(len(ds["train"]))
val_ds = load_dataset("json", data_files={"validation": DATA_DIR + f"/{DATASET_NAME}/valid.jsonl"})
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

def get_completion_text(example):
    """Get completion text with or without <think> reasoning based on flag"""
    completion = example["completion"]
    if not INCLUDE_REASONING:
        # Remove <think>...</think> content but keep the rest
        end_tag = "</think>"
        end_pos = completion.find(end_tag)
        if end_pos != -1:
            completion = "\n\n" + end_tag + end_tag + completion[end_pos + len(end_tag):]
    return completion

# TokenBatcher class removed - using regular DataLoader instead

# Use DistributedSampler properly for DDP
train_sampler = DistributedSampler(ds["train"], shuffle=True)
val_sampler = DistributedSampler(val_ds["validation"], shuffle=False)

def collate_fn(batch):
    texts = [ex["prompt"] + get_completion_text(ex) for ex in batch]
    # Calculate max length in current batch instead of using fixed MAX_SEQ_LEN
    batch_max_len = min(256, max(len(tok.encode(text)) for text in texts))
    # print(f"Batch max length: {batch_max_len}")
    return tok(texts, max_length=batch_max_len, padding=True, truncation=True, return_tensors="pt")

train_dl = DataLoader(ds["train"], batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
val_dl = DataLoader(val_ds["validation"], batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    step_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
    if not step_dirs:
        return None, 0
    steps = [(int(d.split("_")[1]), d) for d in step_dirs if d.split("_")[1].isdigit()]
    if not steps:
        return None, 0
    latest_step, latest_dir = max(steps)
    return os.path.join(checkpoint_dir, latest_dir), latest_step

def load_training_state(checkpoint_dir, optimizer, scheduler):
    state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    if not os.path.exists(state_path):
        return 0, [], [], []
    state = torch.load(state_path, map_location='cuda')
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    return (state.get('step', 0), 
            state.get('train_losses', []), 
            state.get('eval_losses', []), 
            state.get('loss_steps', []))

latest_checkpoint, resume_step = find_latest_checkpoint(CHECKPOINT_DIR) if RESUME else (None, 0)

# Model - Resume from your checkpoint
model_args = {
    "pretrained_model_name_or_path": MODEL,
    "quantization_config": bnb_config,
    "dtype": torch.float16,
    "device_map": {"": local_rank}
}
if latest_checkpoint:
    if logger: logger.info(f"Resuming from step {resume_step}: {latest_checkpoint}")
    base_model = AutoModelForCausalLM.from_pretrained(**model_args)
    if USE_8BIT: base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, latest_checkpoint)
    # Minimal fix: set LoRA params requires_grad=True before DDP
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
else:
    if logger: logger.info(f"Starting fresh training with {'8-bit' if USE_8BIT else '16-bit'} quantization")
    base_model = AutoModelForCausalLM.from_pretrained(**model_args)
    if USE_8BIT: base_model = prepare_model_for_kbit_training(base_model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.02, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora)
    resume_step = 0

model.config.use_cache = False
model.train()

model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
# Minimal fix: Enable input grads and LoRA param grads after DDP wrapping
if hasattr(model, 'module'):
    model.module.enable_input_require_grads()
    for name, param in model.module.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.module.gradient_checkpointing_enable()
else:
    model.enable_input_require_grads()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
sched = get_cosine_with_min_lr_schedule_with_warmup(
    opt, 
    num_warmup_steps=WARMUP, 
    num_training_steps=STEPS, 
    min_lr=MIN_LR  # You can adjust min_lr as needed
)

# Load training state if resuming
if latest_checkpoint:
    resume_step, train_losses, eval_losses, loss_steps = load_training_state(latest_checkpoint, opt, sched)
    if logger: logger.info(f"Resumed optimizer/scheduler at step {resume_step}, LR: {sched.get_last_lr()[0]:.2e}")
else:
    if logger: logger.info(f"Fresh training - Initial LR: {BASE_LR:.2e}")

# Initialize loss tracking (will be overwritten if resuming)
train_losses = []
eval_losses = []
loss_steps = []

# Train

step = resume_step + 1
if logger: logger.info(f"Starting training from step {step} to {STEPS}")

# Create infinite iterator from distributed dataloader
def infinite_dataloader(dataloader):
    while True:
        dataloader.sampler.set_epoch(step // len(dataloader))  # Shuffle each epoch
        for batch in dataloader:
            yield batch

data_iter = infinite_dataloader(train_dl)

try:
    for batch in data_iter:
        if step >= STEPS: break
        input_ids = batch["input_ids"].cuda()
        loss = model(input_ids=input_ids, labels=input_ids).loss / ACCUM; loss.backward()
        if (step + 1) % ACCUM == 0: opt.step(); sched.step(); opt.zero_grad()
        if logger and step % LOG_STEP == 0:
            train_losses.append(loss.item() * ACCUM); loss_steps.append(step)
            logger.info(f"Step {step}/{STEPS} ({(step / STEPS) * 100:.1f}%), Loss: {loss.item() * ACCUM:.4f}, LR: {sched.get_last_lr()[0]:.2e}, Tokens: {input_ids.ne(tok.pad_token_id).sum().item()}")
            logger.info(f"  Scheduler step: {sched.last_epoch}, Optimizer step count: {step}")

        # Evaluation every EVAL_STEP steps
        if step > 0 and step % EVAL_STEP == 0 and local_rank == 0:
            model.eval()
            torch.cuda.empty_cache()
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                with torch.inference_mode():
                    val_loss = sum(model(input_ids=val_batch["input_ids"].cuda(), labels=val_batch["input_ids"].cuda()).loss.item()
                                for val_steps, val_batch in enumerate(val_dl) if val_steps < 5)
                    avg_val_loss = val_loss / 5
                    eval_losses.append(avg_val_loss)
                    if avg_val_loss < best_val_loss:
                        best_val_loss, no_improve_count = avg_val_loss, 0
                        if logger: logger.info(f"✓ New best validation loss: {avg_val_loss:.6f}")
                    else:
                        no_improve_count += 1
                        if logger: logger.info(f"Validation loss at step {step}: {avg_val_loss:.6f} (no improvement: {no_improve_count}/{EARLY_STOP_PATIENCE})")
                        if no_improve_count >= EARLY_STOP_PATIENCE:
                            if logger: logger.info(f"Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} evaluations.")
                            break
            model.train()
        
        # Save checkpoint
        if step > 0 and step % SAVE_STEP == 0 and local_rank == 0:
            d = f"{CHECKPOINT_DIR}/step_{step}"
            (model.module if hasattr(model, 'module') else model).save_pretrained(d)
            torch.save({'step': step, 'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': sched.state_dict(), 'train_losses': train_losses, 'eval_losses': eval_losses, 'loss_steps': loss_steps}, f"{d}/training_state.pt")
            if logger: logger.info(f"✓ Checkpoint saved at step {step}")
        step += 1
except Exception as e:
    if logger:
        logger.error(f"Exception occurred during training: {repr(e)}", exc_info=True)
    else:
        print(f"Exception occurred during training: {repr(e)}")

# dist.destroy_process_group()

if local_rank == 0:
    model.module.save_pretrained(f"{CHECKPOINT_DIR}/final") if hasattr(model, 'module') else model.save_pretrained(f"{CHECKPOINT_DIR}/final")
    plot_losses(train_losses, eval_losses, adapter_name)
    # from pytorchInference import main as inference_main
    # inference_main()