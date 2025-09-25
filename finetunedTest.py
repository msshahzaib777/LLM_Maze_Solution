# --------------------------
# Eval on test.jsonl + save per-example outputs
# --------------------------
import os, re, json
from mlx_lm import generate, load

ds_dir = "data/custom_2"
adapter_dir = "finetuned_model/adapters_available2"
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"

ALLOWED_DIRS = {"up", "down", "left", "right"}
test_path = os.path.join(ds_dir, "test.jsonl")

eval_dir = os.path.join(adapter_dir, "eval_2")
os.makedirs(eval_dir, exist_ok=True)
preds_jsonl = os.path.join(eval_dir, "test_predictions.jsonl")
summary_json = os.path.join(eval_dir, "summary.json")

model_lora, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True}, adapter_path=adapter_dir)  # tokenizer has .apply_chat_template and .encode/.decode


def _extract_first_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*?\}", s, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def _norm_dirs(dirs):
    if not isinstance(dirs, (list, tuple)): return set()
    return {str(x).strip().lower() for x in dirs if str(x).strip().lower() in ALLOWED_DIRS}

total = correct_start = correct_dirs = exact_both = bad_json = 0
tp = fp = fn = 0  # micro P/R/F1 for directions

with open(test_path, "r", encoding="utf-8") as fin, open(preds_jsonl, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        ex = json.loads(line)
        user_prompt = ex["prompt"]
        gt = json.loads(ex["completion"])  # {"start":[r,c], "available_directions":[...]}
        gt_start = [int(gt["start"][0]), int(gt["start"][1])]
        gt_dirs  = _norm_dirs(gt.get("available_directions", []))

        # Compose chat prompt and decode deterministically
        messages = [{"role": "user", "content": user_prompt}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = generate(model_lora, tokenizer, prompt=chat_prompt, max_tokens=64, verbose=False)

        pred = _extract_first_json(out)
        record = {
            "id": i,
            "prompt": user_prompt,
            "ground_truth": {"start": gt_start, "available_directions": sorted(gt_dirs)},
            "raw_output": out,
            "parsed": pred,
        }

        if not pred or "start" not in pred:
            bad_json += 1
            record["prediction"] = None
            record["match"] = {"start": False, "available_directions": False, "both": False}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            continue

        # Parse prediction
        try:
            pred_start = [int(pred["start"][0]), int(pred["start"][1])]
        except Exception:
            bad_json += 1
            record["prediction"] = None
            record["match"] = {"start": False, "available_directions": False, "both": False}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            continue

        pred_dirs = _norm_dirs(pred.get("available_directions", []))
        start_ok = (pred_start == gt_start)
        dirs_ok  = (pred_dirs == gt_dirs)

        # Update set metrics
        tp += len(pred_dirs & gt_dirs)
        fp += len(pred_dirs - gt_dirs)
        fn += len(gt_dirs - pred_dirs)

        record["prediction"] = {"start": pred_start, "available_directions": sorted(pred_dirs)}
        record["match"] = {"start": start_ok, "available_directions": dirs_ok, "both": start_ok and dirs_ok}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        total += 1
        if start_ok: correct_start += 1
        if dirs_ok:  correct_dirs  += 1
        if start_ok and dirs_ok: exact_both += 1

# Aggregate metrics
acc_start = 0.0 if total == 0 else correct_start / total
acc_dirs  = 0.0 if total == 0 else correct_dirs  / total
acc_both  = 0.0 if total == 0 else exact_both    / total
valid_json_rate = 0.0 if total == 0 else 1.0 - (bad_json / total)
precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
recall    = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
f1        = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

summary = {
    "total": total,
    "start_accuracy": acc_start,
    "dirs_accuracy": acc_dirs,
    "exact_both_accuracy": acc_both,
    "dirs_micro_precision": precision,
    "dirs_micro_recall": recall,
    "dirs_micro_f1": f1,
    "valid_json_rate": valid_json_rate,
    "invalid_json": bad_json,
    "preds_file": preds_jsonl,
}
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\nSaved predictions to:", preds_jsonl)
print("Saved summary to    :", summary_json)
print(
    f"\nStart acc={acc_start:.2%} | Dirs acc={acc_dirs:.2%} | Both={acc_both:.2%} | "
    f"P={precision:.2%} R={recall:.2%} F1={f1:.2%} | Valid JSON={valid_json_rate:.2%}"
)
