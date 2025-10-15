import torch, math
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Updated constants for Mac MPS
MODEL = "Qwen/Qwen3-4B"  # Updated to Qwen3
DATA  = "data/custom_curriculum_1/train.jsonl"
BATCH = 1  # Reduced batch size for MPS memory constraints
ACCUM = 8
LR    = 2e-4
EPOCHS= 2
MAXLEN= 512
DEVICE= "mps"  # Force MPS device for Mac

# Check MPS availability
assert torch.backends.mps.is_available(), "MPS not available. Make sure you're on MacOS 12.3+"

ds = load_dataset("json", data_files={"train": DATA})

def format_example(ex):
    return {"text": ex["prompt"] + ex["completion"]}
ds = ds.map(format_example, remove_columns=ds["train"].column_names)

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=MAXLEN)
ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

dl = DataLoader(ds_tok["train"], batch_size=BATCH, shuffle=True, collate_fn=lambda x: {
    "input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(s["input_ids"]) for s in x], batch_first=True, padding_value=tok.pad_token_id),
    "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.tensor(s["attention_mask"]) for s in x], batch_first=True, padding_value=0),
})

# Load model with float16 for MPS
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Configure LoRA
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)
model.to(DEVICE)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=LR)
num_steps = EPOCHS * math.ceil(len(dl))
sched = get_cosine_schedule_with_warmup(opt, int(0.03*num_steps), num_steps)

# Training loop
global_step = 0
opt.zero_grad()
for epoch in range(EPOCHS):
    for batch in dl:
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
            
        if global_step % 10 == 0:  # Print loss every 10 steps
            print(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()*ACCUM}")
            
        global_step += 1
        
    # Save checkpoint after each epoch
    model.save_pretrained(f"finetuned_model/adapters_dir_qwen3/epoch_{epoch+1}")

# Final save
model.save_pretrained("finetuned_model/adapters_dir_qwen3/final")
tok.save_pretrained("finetuned_model/adapters_dir_qwen3/final")
