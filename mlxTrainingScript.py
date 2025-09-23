# -*- coding: utf-8 -*-
import json, os, math, pathlib, random
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, train  # we'll use our own dataset class
from classes.MazeJSONLDataset import MazeJSONLDataset
from mlx_lm.tuner import linear_to_lora_layers
from classes.metrics import SimpleMetrics
# NOTE: we don't rely on mlx_lm.tuner.datasets to avoid format mismatch

# --------------------------
# Config
# --------------------------
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
adapter_dir = "finetuned_model/adapters"
# --------------------------
# Datasets
# --------------------------
# Point these to your prepared JSONL files.
# Tip: keep 10% val, stratify by maze size (3..7).
train_jsonl = "data/maze_training_train.json"
val_jsonl   = "data/maze_training_test.json"

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
ITERS = 3000       # ~few epochs over 25–50k rows; adjust to your dataset size
EVAL_EVERY = 200

# --------------------------
# Load model + tokenizer
# --------------------------
model, tokenizer = load(model_path)  # tokenizer has .apply_chat_template and .encode/.decode

# Quick sanity generation
messages = [{"role": "user", "content": "You will be fine-tuned to read ASCII mazes."}]
preamble = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
_ = generate(model, tokenizer, prompt=preamble, max_tokens=32, verbose=False)

# --------------------------
# Dataset: JSONL → supervised pairs
# --------------------------
def build_prompt(maze_ascii: str, user_prompt: str) -> str:
    # Keep formatting consistent so tokenization is stable.
    user = (
        "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
        "Return only these keys: `start` (0-based [row,col]) and `available_directions` "
        "(array using 'up','down','left','right').\n\n"
        "<maze>\n" + maze_ascii + "\n</maze>\n\n"
        + user_prompt.strip()
    )
    messages = [
        {"role": "system", "content": "Follow the schema exactly. No extra text."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_target(answer_start: List[int], answer_dirs: List[str]) -> str:
    # The trainer will learn only from these target tokens (prompt is masked out).
    return json.dumps({
        "start": [int(answer_start[0]), int(answer_start[1])],
        "available_directions": list(answer_dirs)
    }, ensure_ascii=False)

def clamp_and_pad(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) > max_len:
        # For chat SFT, truncating the left (prompt side) is usually safer than chopping off the label.
        # But since we create the full sequence ourselves, keep it simple: right-truncate.
        ids = ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))

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
train_set = MazeJSONLDataset(train_jsonl, tokenizer, max_seq_len=MAX_SEQ_LEN, target_mode="start_and_dirs")
val_set   = MazeJSONLDataset(val_jsonl,   tokenizer, max_seq_len=MAX_SEQ_LEN, target_mode="start_and_dirs")

print(f"Loaded train: {len(train_set)}, val: {len(val_set)}")

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
    tokenizer=tokenizer,
    args=training_args,
    optimizer=optimizer,
    train_dataset=train_set,
    val_dataset=val_set,
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
