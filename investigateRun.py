# --------------------------
# Eval on test.jsonl + save per-example outputs
# --------------------------
import os, re, json
from mlx_lm import generate, load

def _norm_dirs(dirs):
    if not isinstance(dirs, (list, tuple)): return set()
    return {str(x).strip().lower() for x in dirs if str(x).strip().lower() in ALLOWED_DIRS}

ds_dir = "data/custom_2"
adapter_dir = "finetuned_model/adapters_available"
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"

ALLOWED_DIRS = {"up", "down", "left", "right"}
test_path = os.path.join(ds_dir, "test.jsonl")

model_lora, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True}, adapter_path=adapter_dir)  # tokenizer has .apply_chat_template and .encode/.decode

#dataset Generation

from Dataset_Gen.utils import dict_to_prompt_completion, build_prompt, build_target
from Dataset_Gen.utils import make_training_example
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import json, os

split_name = "2"

filename = f'../data/maze_training_{split_name}.json'
dataset_dir = f'../data/custom_{split_name}'

os.makedirs("../data", exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

target_mode = "start_available_direction" # start_only, optimal_next_step.

# --- Define sizes and counts ---
size_counts = {
    3: 1500,   # 1000 mazes of size 5x5
    4: 2500,       # 500 mazes of size 7x7
    5: 3500,       # 250 mazes of size 9x9
    6: 5000,
    7: 7000,      # 100 mazes of size 11x11
}

g = 5
m = Maze()
m.generator = Prims(g, g)
m.solver = BacktrackingSolver()
m.generate()
m.generate_entrances(start_outer=False, end_outer=False)
training_example = make_training_example(m)
ex_jsonl = dict_to_prompt_completion(training_example, target_mode)
ex = json.loads(ex_jsonl)
user_prompt = ex["prompt"]
gt = json.loads(ex["completion"])  # {"start":[r,c], "available_directions":[...]}
gt_start = [int(gt["start"][0]), int(gt["start"][1])]
gt_dirs  = _norm_dirs(gt.get("available_directions", []))

# Compose chat prompt and decode deterministically
messages = [{"role": "user", "content": user_prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = generate(model_lora, tokenizer, prompt=chat_prompt, max_tokens=64, verbose=False)

print(out)
print(gt)

print(m)