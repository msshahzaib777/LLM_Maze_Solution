# --------------------------
# Batched eval on test.jsonl + save per-example outputs
# --------------------------
import os, re, json, math
from typing import List
from mlx_lm import generate, load

# --------------------------
# Paths & constants
# --------------------------
ds_dir = "data/custom__start_and_end_1"
adapter_dir = "finetuned_model/adapters_dir_start_end"
model_path = "Qwen/Qwen3-4B-MLX-bf16"

ALLOWED_DIRS = {"up", "down", "left", "right"}
test_path = os.path.join(ds_dir, "test.jsonl")

eval_dir = os.path.join(adapter_dir, "eval_1")
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

def _evaluate_from_prediction_records(pred_records_iter):
    """
    Recompute metrics from a stream of per-example prediction dicts
    (each line previously written by this script).

    Returns: summary dict
    """
    total = correct_start = correct_dirs = exact_all = bad_json = correct_end = 0
    tp = fp = fn = 0

    for rec in pred_records_iter:
        total += 1

        # Prefer explicit "prediction"; otherwise try "parsed"
        pred = rec.get("prediction")
        if pred is None:
            parsed = rec.get("parsed")
            if not parsed:
                bad_json += 1
                continue
            pred = {}
            try:
                if "start" in parsed and "end" in parsed:
                    pred["start"] = [int(parsed["start"][0]), int(parsed["start"][1])]
                    pred["end"]   = [int(parsed["end"][0]),   int(parsed["end"][1])]
                else:
                    bad_json += 1
                    continue
                pred["available_directions"] = parsed.get("available_directions", [])
            except Exception:
                bad_json += 1
                continue

        gt = rec["ground_truth"]
        gt_start = [int(gt["start"][0]), int(gt["start"][1])]
        gt_end   = [int(gt["end"][0]),   int(gt["end"][1])]
        gt_dirs  = _norm_dirs(gt.get("available_directions", []))

        try:
            pred_start = [int(pred["start"][0]), int(pred["start"][1])]
            pred_end   = [int(pred["end"][0]),   int(pred["end"][1])]
        except Exception:
            bad_json += 1
            continue

        pred_dirs = _norm_dirs(pred.get("available_directions", []))

        start_ok = (pred_start == gt_start)
        end_ok   = (pred_end   == gt_end)
        dirs_ok  = (pred_dirs  == gt_dirs)

        # Set metrics for directions
        tp += len(pred_dirs & gt_dirs)
        fp += len(pred_dirs - gt_dirs)
        fn += len(gt_dirs - pred_dirs)

        if start_ok: correct_start += 1
        if end_ok:   correct_end   += 1
        if dirs_ok:  correct_dirs  += 1
        if start_ok and end_ok and dirs_ok: exact_all += 1

    acc_start = 0.0 if total == 0 else correct_start / total
    acc_end   = 0.0 if total == 0 else correct_end   / total
    acc_dirs  = 0.0 if total == 0 else correct_dirs  / total
    acc_all   = 0.0 if total == 0 else exact_all     / total
    valid_json_rate = 0.0 if total == 0 else 1.0 - (bad_json / total)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1        = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "total": total,
        "start_accuracy": acc_start,
        "end_accuracy": acc_end,
        "dirs_accuracy": acc_dirs,
        "exact_both_accuracy": acc_all,
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

# --------------------------
# Load (possibly) the model — but only if needed
# --------------------------
def _lazy_load_model():
    global model_lora, tokenizer
    if "model_lora" in globals() and "tokenizer" in globals():
        return
    model, tok = load(
        model_path,
        tokenizer_config={"trust_remote_code": True},
        adapter_path=adapter_dir
    )
    model_lora, tokenizer = model, tok

# --------------------------
# Read dataset into memory (single pass I/O)
# --------------------------
records = []
with open(test_path, "r", encoding="utf-8") as fin:
    for line in fin:
        ex = json.loads(line)
        user_prompt = ex["prompt"]
        gt = json.loads(ex["completion"])  # {"start":[r,c], "available_directions":[...], "end":[r,c]}
        gt_start = [int(gt["start"][0]), int(gt["start"][1])]
        gt_end   = [int(gt["end"][0]),   int(gt["end"][1])]
        gt_dirs  = _norm_dirs(gt.get("available_directions", []))
        # Delay tokenizer usage until/if we actually infer
        records.append({
            "prompt_raw": user_prompt,
            "gt_start": gt_start,
            "gt_dirs": gt_dirs,
            "gt_end": gt_end,
        })

# --------------------------
# Decide: use cached predictions or run inference
# --------------------------
use_cached = os.path.exists(preds_jsonl) and os.path.getsize(preds_jsonl) > 0

if use_cached:
    print(f"Using cached predictions from: {preds_jsonl}")

    # Evaluate straight from cached file
    with open(preds_jsonl, "r", encoding="utf-8") as fin:
        pred_records = (json.loads(line) for line in fin)

        summary = _evaluate_from_prediction_records(pred_records)

else:
    print("No cached predictions found — running inference and writing cache...")

    # Need tokenizer to build chat prompts only if we infer
    _lazy_load_model()

    # Build chat prompts now
    for r in records:
        r["chat_prompt"] = _build_chat_prompt(r["prompt_raw"])

    # Inference + write predictions as we go, and accumulate for metrics
    total = correct_start = correct_dirs = exact_all = bad_json = correct_end = 0
    tp = fp = fn = 0

    with open(preds_jsonl, "w", encoding="utf-8") as fout:
        for b in range(0, len(records), BATCH_SIZE):
            batch = records[b:b+BATCH_SIZE]
            chat_prompts = [r["chat_prompt"] for r in batch]

            outs = _batched_generate(chat_prompts)
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
                        "available_directions": sorted(rec["gt_dirs"]),
                        "end": rec["gt_end"],
                    },
                    "raw_output": out,
                    "parsed": pred,
                }

                if not pred or "start" not in pred or "end" not in pred:
                    bad_json += 1
                    record["prediction"] = None
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    continue

                # Parse prediction
                try:
                    pred_start = [int(pred["start"][0]), int(pred["start"][1])]
                    pred_end   = [int(pred["end"][0]),   int(pred["end"][1])]
                except Exception:
                    bad_json += 1
                    record["prediction"] = None
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    continue

                pred_dirs = _norm_dirs(pred.get("available_directions", []))
                start_ok = (pred_start == rec["gt_start"])
                end_ok   = (pred_end   == rec["gt_end"])
                dirs_ok  = (pred_dirs  == rec["gt_dirs"])

                # Update set metrics
                tp += len(pred_dirs & rec["gt_dirs"])
                fp += len(pred_dirs - rec["gt_dirs"])
                fn += len(rec["gt_dirs"] - pred_dirs)

                record["prediction"] = {
                    "start": pred_start,
                    "available_directions": sorted(pred_dirs),
                    "end": pred_end,
                }
                record["match"] = {
                    "start": start_ok,
                    "available_directions": dirs_ok,
                    "end": end_ok,
                    "all": start_ok and dirs_ok and end_ok,
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                total += 1
                if start_ok: correct_start += 1
                if end_ok:   correct_end   += 1
                if dirs_ok:  correct_dirs  += 1
                if start_ok and end_ok and dirs_ok: exact_all += 1

    acc_start = 0.0 if total == 0 else correct_start / total
    acc_end   = 0.0 if total == 0 else correct_end   / total
    acc_dirs  = 0.0 if total == 0 else correct_dirs  / total
    acc_all   = 0.0 if total == 0 else exact_all     / total
    valid_json_rate = 0.0 if total == 0 else 1.0 - (bad_json / total)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1        = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    summary = {
        "total": total,
        "start_accuracy": acc_start,
        "end_accuracy": acc_end,
        "dirs_accuracy": acc_dirs,
        "exact_both_accuracy": acc_all,
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

# --------------------------
# Save summary + print
# --------------------------
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved predictions to:", preds_jsonl)
print("Saved summary to    :", summary_json)
print(
    f"\nStart acc={summary['start_accuracy']:.2%} | "
    f"Dirs acc={summary['dirs_accuracy']:.2%} | "
    f"Both={summary['exact_both_accuracy']:.2%} | "
    f"P={summary['dirs_micro_precision']:.2%} "
    f"R={summary['dirs_micro_recall']:.2%} "
    f"F1={summary['dirs_micro_f1']:.2%} | "
    f"Valid JSON={summary['valid_json_rate']:.2%}"
)