import torch, math
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-7B"
DATA  = "data/train.jsonl"
BATCH = 2
ACCUM = 8
LR    = 2e-4
EPOCHS= 2
MAXLEN= 2048
DEVICE= "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.to(DEVICE)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=LR)
num_steps = EPOCHS * math.ceil(len(dl))
sched = get_cosine_schedule_with_warmup(opt, int(0.03*num_steps), num_steps)

global_step = 0
opt.zero_grad()
for epoch in range(EPOCHS):
    for batch in dl:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        # causal labels = input_ids shifted inside the model; pass labels=input_ids
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = out.loss / ACCUM
        loss.backward()
        if (global_step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        global_step += 1
    # save per-epoch
    model.save_pretrained(f"outputs/lora_adapter_epoch{epoch+1}")

# final save
model.save_pretrained("outputs/lora_adapter")
tok.save_pretrained("outputs/lora_adapter")
