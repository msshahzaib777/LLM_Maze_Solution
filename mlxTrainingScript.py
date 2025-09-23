# -*- coding: utf-8 -*-
import json, os, math, pathlib, random
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, train  # we'll use our own dataset class
# NOTE: we don't rely on mlx_lm.tuner.datasets to avoid format mismatch

# --------------------------
# Config
# --------------------------
# Use your actual model id/path; you said "Qwen3 4B Thinking 2507 bf16 mlx (~8GB)"
# Example placeholders you can swap:
# model_path = "mlx-community/Qwen2.5-4B-Instruct-MLX"
# or your local fused weights directory:
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
adapter_dir = "finetuned_model/adapters"
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

class MazeJSONLDataset:
    """
    Expects a JSONL file with rows like:
    {
      "maze": "#########\\n#   S  E\\n#########",
      "prompt": "Identify the start location and its available directions in this maze.",
      "answer": {"start":[r,c], "available_directions":["left","up"] }
      // optional: "chain_of_thought", "solved_maze"
    }
    Emits dicts with 'input_ids' and 'labels' (prompt masked with -100).
    """
    def __init__(self,
                 jsonl_path: str,
                 tokenizer,
                 max_seq_len: int = 256,
                 target_mode: str = "start_and_dirs"  # or "start_only"
                 ):
        self.items: List[Dict[str, Any]] = []
        self.tok = tokenizer
        self.max_len = max_seq_len
        self.pad_id = tokenizer.pad_id if hasattr(tokenizer, "pad_id") and tokenizer.pad_id is not None else 0
        self.target_mode = target_mode

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                maze = ex["maze"]
                user_prompt = ex.get("prompt", "Identify the start location and its available directions in this maze.")
                # Ground truth:
                ans = ex.get("answer", {})
                start = ans.get("start", None)
                dirs  = ans.get("available_directions", None)

                # Build strings
                prompt_text = build_prompt(maze, user_prompt)

                if self.target_mode == "start_only":
                    target_obj = {"start": start}
                    target_text = json.dumps(target_obj, ensure_ascii=False)
                else:
                    target_text = build_target(start, dirs)

                # Tokenize full sequence = [prompt][assistant target]
                prompt_ids = self.tok.encode(prompt_text)
                target_ids = self.tok.encode(target_text)

                input_ids = prompt_ids + target_ids
                input_ids = clamp_and_pad(input_ids, self.max_len, self.pad_id)

                # Labels: mask prompt part (set to -100), keep target tokens
                labels = [-100] * min(len(prompt_ids), self.max_len)
                tail = self.max_len - len(labels)
                if tail > 0:
                    # the tail corresponds to (part of) target_ids (padded to max_len)
                    tgt_tail = clamp_and_pad(target_ids, tail, self.pad_id)
                    # masked positions for padding should also be -100 (so they don’t contribute to loss)
                    labels += [tid if tid != self.pad_id else -100 for tid in tgt_tail]
                else:
                    labels = labels[:self.max_len]

                self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

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
from mlx_lm.tuner import linear_to_lora_layers
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Trainable params (LoRA): {num_train_params:,}")

model.train()

# --------------------------
# Datasets
# --------------------------
# Point these to your prepared JSONL files.
# Tip: keep 10% val, stratify by maze size (3..7).
train_jsonl = "data/mazes_stageA_train.jsonl"
val_jsonl   = "data/mazes_stageA_val.jsonl"

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
class SimpleMetrics:
    def __init__(self):
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses:   List[Tuple[int, float]] = []
    def on_train_step_end(self, it: int, loss: float):
        self.train_losses.append((it, float(loss)))
    def on_eval_end(self, it: int, loss: float):
        self.val_losses.append((it, float(loss)))

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
