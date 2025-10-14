import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen2.5-7B"
adapter = "outputs/lora_adapter"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model.eval()

prompt = "Maze:\n#####\n#S  #\n# ##E\n#####\nNext move?"
inp = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
