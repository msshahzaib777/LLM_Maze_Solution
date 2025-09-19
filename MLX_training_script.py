import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.dora import add_dora   # DoRA support
from mlx.data import DataLoader

# --- Config ---
model_id = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
batch_size = 2
lr = 5e-5
epochs = 3
max_len = 512

# --- Load pretrained model + tokenizer ---
model, tokenizer = load(model_id)

# Inject DoRA adapters into target modules
# Typical targets: attention projections + MLP projections
model = add_dora(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "up_proj", "down_proj", "gate_proj"      # MLP
    ],
    rank=16,    # adapter rank
    alpha=32,   # scaling factor
    dropout=0.05
)

# --- Prepare dataset ---
with open("Dataset_Gen/maze_training.json") as f:
    raw = json.load(f)

pairs = []
for ex in raw:
    maze = ex["maze"]
    prompt = ex["prompt"]
    answer = json.dumps(ex["answer"])  # structured JSON string

    input_text = f"Maze:\n{maze}\n\nTask: {prompt}\nAnswer:"
    output_text = f" {answer}"

    enc_in = tokenizer(input_text, truncation=True, max_length=max_len, return_tensors="np")
    enc_out = tokenizer(output_text, truncation=True, max_length=max_len, return_tensors="np")

    input_ids = enc_in["input_ids"][0].tolist()
    label_ids = enc_out["input_ids"][0].tolist()

    full_ids = input_ids + label_ids
    labels = [-100] * len(input_ids) + label_ids  # mask out prompt

    pairs.append({"input_ids": full_ids, "labels": labels})

# --- Collate function ---
def collate(batch):
    max_len_batch = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    for x in batch:
        ids = x["input_ids"]
        labs = x["labels"]
        pad_len = max_len_batch - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
    return {
        "input_ids": mx.array(input_ids, dtype=mx.int32),
        "labels": mx.array(labels, dtype=mx.int32),
    }

loader = DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=collate)

# --- Optimizer ---
opt = optim.AdamW(model.parameters(), lr=lr)

# --- Loss ---
loss_fn = nn.losses.cross_entropy

# --- Training loop ---
for epoch in range(epochs):
    total_loss = 0
    for step, batch in enumerate(loader):
        x = batch["input_ids"]
        y = batch["labels"]

        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = mx.mean(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    print(f"Epoch {epoch} avg loss {total_loss / len(loader):.4f}")

# --- Save adapters only ---
model.save_dora("qwen3_maze_dora")
print("✅ Training done, DoRA adapters saved to qwen3_maze_dora/")