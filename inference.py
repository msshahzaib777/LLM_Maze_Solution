from Dataset_Gen.utils import build_prompt
from mlx_lm import generate, load

# --------------------------
# Load with adapter & test
# --------------------------
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
adapter_dir = "finetuned_model/adapters"
ds_dir = "data/custom_1"

model_lora, tokenizer = load(model_path, adapter_path=adapter_dir)

# # Tiny sanity check on a single sample prompt
# demo_maze = (
#     "###########\n"
#     "# # # # # #\n"
#     "###########\n"
#     "# # # # # #\n"
#     "###########\n"
#     "#     #  E#\n"
#     "# ### ### #\n"
#     "# # #   # #\n"
#     "# ####### #\n"
#     "#        S#\n"
#     "###########"
# )
# demo_prompt = "Identify the start location and its available directions in this maze."
# demo_infer = build_prompt(demo_maze, demo_prompt)
# resp = generate(model_lora, tokenizer, prompt=demo_infer, max_tokens=64, verbose=True)
# print("\nMODEL OUTPUT:\n", resp)


# --------------------------
# Eval on test.jsonl: Start-index accuracy
# --------------------------
import re, json

test_path = os.path.join(ds_dir, "test.jsonl")

def _extract_first_json(s: str):
    # try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    # fallback: grab first {...} block
    m = re.search(r"\{.*?\}", s, flags=re.S)
    if not m:
        return None
    chunk = m.group(0).strip()
    try:
        return json.loads(chunk)
    except Exception:
        return None

total = 0
correct = 0
bad_json = 0

with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        user_prompt = ex["prompt"]
        gt = json.loads(ex["completion"])  # expected: {"start":[r,c], "available_directions":[...]}
        gt_start = list(map(int, gt.get("start", [])))

        # compose chat prompt from plain prompt text
        messages = [{"role": "user", "content": user_prompt}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # deterministic decode for accuracy
        out = generate(
            model_lora, tokenizer,
            prompt=chat_prompt,
            max_tokens=64,
            temperature=0.0,
            verbose=False
        )

        pred = _extract_first_json(out)
        if not pred or "start" not in pred:
            bad_json += 1
            total += 1
            continue

        pred_start = list(map(int, pred["start"]))
        if pred_start == gt_start:
            correct += 1
        total += 1

acc = 0.0 if total == 0 else correct / total
print(f"\nStart index accuracy: {acc:.2%}  (correct={correct}/{total}, invalid_json={bad_json})")
