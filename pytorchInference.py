import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

base_id = "Qwen/Qwen3-4B"          # example
adapter_path = "./finetuned_model/adapter/adapters_merged_2"     # PEFT-style adapter

tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto")
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

test_file = "data/custom_curriculum_1/test.jsonl"
# Load one example with prompt/completion pair from test file
with open(test_file, 'r') as f:
    test_example = json.loads(f.readline())

# Format prompt from the example
prompt = test_example['prompt']
inp = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
