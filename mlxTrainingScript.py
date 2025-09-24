# -*- coding: utf-8 -*-
import json, os
import types

import matplotlib.pyplot as plt

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, train  # we'll use our own dataset class

from Dataset_Gen.utils import build_prompt
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.datasets import load_dataset, CacheDataset
from classes.metrics import SimpleMetrics

# NOTE: we don't rely on mlx_lm.tuner.datasets to avoid format mismatch

# --------------------------
# Config
# --------------------------
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
adapter_dir = "finetuned_model/adapters_available"
# --------------------------
# Datasets
# --------------------------
# Point these to your prepared JSONL files.
# Tip: keep 10% val, stratify by maze size (3..7).
ds_dir = "data/custom_2"

os.makedirs(adapter_dir, exist_ok=True)
adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
adapter_file_path   = os.path.join(adapter_dir, "adapters.safetensors")

# LoRA settings matched to your 4B & M2 Max 64GB
lora_config = {
    "num_layers": 12,  # LoRA on last 12 blocks is a solid start for 4B
    "lora_parameters": {
        "rank": 16,    # r=16 is a good default; try 8 if memory-constrained, 32 if underfitting
        "scale": 16.0, # α typically = r (or 2r). Start with r.
        "dropout": 0.05,
    },
}

# Training settings
MAX_SEQ_LEN = 256  # enough for 7x7 grid + prompt + short JSON answer
LR = 1.0e-4        # LoRA LR (cosine schedule handled inside trainer if available)
ITERS = 240       # ~few epochs over 25–50k rows; adjust to your dataset size
EVAL_EVERY = 30

# --------------------------
# Load model + tokenizer
# --------------------------
model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})  # tokenizer has .apply_chat_template and .encode/.decode

# Quick sanity generation
messages = [{"role": "user", "content": "You will be fine-tuned to read ASCII mazes."}]
preamble = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
_ = generate(model, tokenizer, prompt=preamble, max_tokens=32, verbose=False)

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
optimizer = optim.Adam(learning_rate=LR)

# Optional: simple callback to collect losses (compatible with your Metrics helper)
metrics = SimpleMetrics()

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

# --------------------------
# Load with adapter & test
# --------------------------
model_lora, _ = load(model_path, adapter_path=adapter_dir)

# Tiny sanity check on a single sample prompt
demo_maze = (
    "###########\n"
    "# # # # # #\n"
    "###########\n"
    "# # # # # #\n"
    "###########\n"
    "#     #  E#\n"
    "# ### ### #\n"
    "# # #   # #\n"
    "# ####### #\n"
    "#        S#\n"
    "###########"
)
demo_prompt = "Identify the start location and its available directions in this maze."
demo_infer = build_prompt(demo_maze, demo_prompt)
resp = generate(model_lora, tokenizer, prompt=demo_infer, max_tokens=64, verbose=True)
print("\nMODEL OUTPUT:\n", resp)
