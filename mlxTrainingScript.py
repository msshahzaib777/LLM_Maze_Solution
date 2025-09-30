# -*- coding: utf-8 -*-
import json, os
import types
import numpy as np
import matplotlib.pyplot as plt

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, train  # we'll use our own dataset class

from Dataset_Gen.utils import build_prompt
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.datasets import load_dataset, CacheDataset
from classes.metrics import LRSchedulerCallback, SimpleMetrics

# NOTE: we don't rely on mlx_lm.tuner.datasets to avoid format mismatch

# --------------------------
# Config
# --------------------------
model_path = "Qwen/Qwen3-4B-MLX-bf16"
adapter_dir = "finetuned_model/adapters_dir_start_4"
# --------------------------
# Datasets
# --------------------------
# Point these to your prepared JSONL files.
# Tip: keep 10% val, stratify by maze size (3..7).
ds_dir = "data/custom_3"

os.makedirs(adapter_dir, exist_ok=True)
adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
adapter_file_path   = os.path.join(adapter_dir, "adapters.safetensors")

# LoRA settings matched to your 4B & M2 Max 64GB
lora_config = {
    "num_layers": 16,  # LoRA on last 12 blocks is a solid start for 4B
    "lora_parameters": {
        "rank": 32,    # r=16 is a good default; try 8 if memory-constrained, 32 if underfitting
        "scale": 32.0, # α typically = r (or 2r). Start with r.
        "dropout": 0.05,
    },
}

# Training settings
MAX_SEQ_LEN = 256  # enough for 7x7 grid + prompt + short JSON answer
BASE_LR = 3.0e-5        # LoRA LR (cosine schedule handled inside trainer if available)
ITERS = 3000       # ~few epochs over 25–50k rows; adjust to your dataset size
WARMUP = int(0.03 * ITERS)
DECAY_STEPS = ITERS - WARMUP
LR_FLOOR = 0.1 * BASE_LR
EVAL_EVERY = 100
TRAINING_CONTINUE = False
# --------------------------
# Load model + tokenizer
# --------------------------

if TRAINING_CONTINUE :
    model, tokenizer = load(model_path, adapter_path= adapter_dir,tokenizer_config={"trust_remote_code": True})
else:
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})  # tokenizer has .apply_chat_template and .encode/.decode
# --------------------------
# Write LoRA adapter config
# --------------------------
with open(adapter_config_path, "w", encoding="utf-8") as f:
    json.dump(lora_config, f, indent=2)

# --------------------------
# Prepare LoRA
# --------------------------
model.freeze()


# Convert the last N linear layers in each block to LoRA (uses mlx_lm.tuner.linear_to_lora_layers internally)
if not TRAINING_CONTINUE :
    linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Trainable params (LoRA): {num_train_params:,}")

model.train()

# --------------------------
# Datasets
# --------------------------
# If you also have “start-only” rows, you can instantiate a second dataset and concatenate.
ds_args = types.SimpleNamespace(
    data=ds_dir,
    train=True,
    test=True,
    # data_format= "completion",
    max_seq_len = MAX_SEQ_LEN,
    mask_prompt=True
)
train_set, val_set, test_set = load_dataset(ds_args, tokenizer)

print(f"Loaded train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
# --------------------------
# Training args & run
# --------------------------
training_args = TrainingArgs(
    adapter_file=adapter_file_path,
    iters=ITERS,
    steps_per_eval=EVAL_EVERY,
    # If your MLX version supports these, uncomment and tune:
    # save_every=1000,
    # log_every=50,
)

# Adam is fine for LoRA; keep LR ~1e-4; weight decay optional for LoRA adapters.
cos = optim.cosine_decay(BASE_LR, ITERS, LR_FLOOR)

optimizer = optim.Adam(learning_rate=cos)

# Optional: simple callback to collect losses (compatible with your Metrics helper)
metrics =  SimpleMetrics()

train(
    model=model,
    args=training_args,
    optimizer=optimizer,
    train_dataset=CacheDataset(train_set),
    val_dataset=CacheDataset(val_set),
    training_callback=metrics,
)

# --------------------------
# Plot losses
# --------------------------
if metrics.train_losses:
    train_its, train_losses = zip(*metrics.train_losses)
    plt.plot(train_its, train_losses, "-o", label="Train")
if metrics.val_losses:
    val_its, val_losses = zip(*metrics.val_losses)
    plt.plot(val_its, val_losses, "-o", label="Validation")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()