# --------------------------
# Batched eval on test.jsonl + save per-example outputs
# --------------------------
import os, re, json, math
from typing import List
from mlx_lm import generate, load

# --------------------------
# Paths & constants
# --------------------------
ds_dir = "data/custom_3"
adapter_dir = "finetuned_model/adapters_dir_start_5"
model_path = "Qwen/Qwen3-4B-MLX-bf16"

ALLOWED_DIRS = {"up", "down", "left", "right"}
test_path = os.path.join(ds_dir, "test.jsonl")

eval_dir = os.path.join(adapter_dir, "eval_2")
os.makedirs(eval_dir, exist_ok=True)
preds_jsonl = os.path.join(eval_dir, "test_predictions.jsonl")
summary_json = os.path.join(eval_dir, "summary.json")

# Tune batch throughput vs. memory
BATCH_SIZE = 64          # try 8–64 depending on VRAM
MAX_TOKENS = 64          # decoding budget per sample
TEMPERATURE = 0.0        # deterministic
TOP_P = 1.0
SEED = 0
VERBOSE = False

# If your mlx_lm version doesn't support batched `prompts=[...]`,
# set this to True to use a small thread pool fallback.
USE_THREADPOOL_FALLBACK = False
THREADS = 4

# --------------------------
# Load model
# --------------------------
model_lora, tokenizer = load(
    model_path,
    tokenizer_config={"trust_remote_code": True},
    adapter_path=adapter_dir
)

# --------------------------
# Utils
# --------------------------
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

def _build_chat_prompt(user_prompt: str) -> str:
    # Wrap as a user message and add generation tag
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _batched_generate(prompts: List[str]) -> List[str]:
    """
    Try true batched generation (vectorized).
    If not supported by your mlx_lm version and fallback is enabled, use threads.
    """
    try:
        # Many mlx_lm builds support a list under the `prompts` argument
        return generate(
            model_lora,
            tokenizer,
            prompts=prompts,
            max_tokens=MAX_TOKENS,
            verbose=VERBOSE,
        )
    except TypeError:
        # Fallback: per-prompt generate (optionally in a small thread pool)
        if not USE_THREADPOOL_FALLBACK:
            return [
                generate(
                    model_lora, tokenizer,
                    prompt=p,
                    max_tokens=MAX_TOKENS,
                    verbose=VERBOSE,
                )
                for p in prompts
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            outs = [None] * len(prompts)
            with ThreadPoolExecutor(max_workers=THREADS) as ex:
                futs = {
                    ex.submit(
                        generate,
                        model_lora, tokenizer,
                        p, MAX_TOKENS, TEMPERATURE, TOP_P, SEED, VERBOSE
                    ): idx
                    for idx, p in enumerate(prompts)
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    outs[idx] = fut.result()
            return outs

# --------------------------
# Read dataset into memory (single pass I/O)
# --------------------------
records = []
with open(test_path, "r", encoding="utf-8") as fin:
    for line in fin:
        ex = json.loads(line)
        user_prompt = ex["prompt"]
        gt = json.loads(ex["completion"])  # {"start":[r,c], "available_directions":[...]}
        gt_start = [int(gt["start"][0]), int(gt["start"][1])]
        gt_dirs  = _norm_dirs(gt.get("available_directions", []))
        chat_prompt = _build_chat_prompt(user_prompt)
        records.append({
            "prompt_raw": user_prompt,
            "chat_prompt": chat_prompt,
            "gt_start": gt_start,
            "gt_dirs": gt_dirs,
        })

# --------------------------
# Batched inference + scoring
# --------------------------
total = correct_start = correct_dirs = exact_both = bad_json = 0
tp = fp = fn = 0  # micro P/R/F1 for directions

with open(preds_jsonl, "w", encoding="utf-8") as fout:
    for b in range(0, len(records), BATCH_SIZE):
        batch = records[b:b+BATCH_SIZE]
        chat_prompts = [r["chat_prompt"] for r in batch]

        outs = _batched_generate(chat_prompts)
        # Normalize to list
        if isinstance(outs, str):
            outs = [outs]

        for j, out in enumerate(outs):
            rec = batch[j]
            i_global = b + j

            pred = _extract_first_json(out)
            record = {
                "id": i_global,
                "prompt": rec["prompt_raw"],
                "ground_truth": {
                    "start": rec["gt_start"],
                    "available_directions": sorted(rec["gt_dirs"])
                },
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
            start_ok = (pred_start == rec["gt_start"])
            dirs_ok  = (pred_dirs == rec["gt_dirs"])

            # Update set metrics
            tp += len(pred_dirs & rec["gt_dirs"])
            fp += len(pred_dirs - rec["gt_dirs"])
            fn += len(rec["gt_dirs"] - pred_dirs)

            record["prediction"] = {
                "start": pred_start,
                "available_directions": sorted(pred_dirs)
            }
            record["match"] = {
                "start": start_ok,
                "available_directions": dirs_ok,
                "both": start_ok and dirs_ok
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += 1
            if start_ok: correct_start += 1
            if dirs_ok:  correct_dirs  += 1
            if start_ok and dirs_ok: exact_both += 1

# --------------------------
# Aggregate metrics
# --------------------------
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
    "batch_size": BATCH_SIZE,
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
}
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved predictions to:", preds_jsonl)
print("Saved summary to    :", summary_json)
print(
    f"\nStart acc={acc_start:.2%} | Dirs acc={acc_dirs:.2%} | Both={acc_both:.2%} | "
    f"P={precision:.2%} R={recall:.2%} F1={f1:.2%} | Valid JSON={valid_json_rate:.2%}"
)
