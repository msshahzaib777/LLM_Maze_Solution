import os
import json
import csv
import re
from mlx_lm import load, generate

def extract_predicted_start(output: str):
    """Extracts [row, col] from model output."""
    match = re.search(r"\[ *(\d+) *, *(\d+) *\]", output)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    return None


def main():
    # --- Path to fused model ---
    fused_model_path = "/Users/studentone/Documents/LLM_distributed Training/Fine-tuned_models/models/qwen3_36_800"

    # --- Load fused model + tokenizer ---
    model, tokenizer = load(fused_model_path)

    # --- Path to test datasets ---
    test_data_dir = "../data/test_data"

    # storage
    all_results = []
    wrong_results = []   # only mistakes

    for filename in sorted(os.listdir(test_data_dir)[:3]):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(test_data_dir, filename)
        print(f"\n🔹 Running inference on {filename}...")

        total = 0
        correct = 0

        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, start=1):
                ex = json.loads(line)

                prompt = ex["prompt"]
                gold = extract_predicted_start(ex["completion"])

                # --- Run inference ---
                output = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=200
                )

                pred = extract_predicted_start(output)
                is_correct = (pred == gold)
                total += 1
                correct += int(is_correct)

                # record all results
                all_results.append({
                    "file": filename,
                    "example": line_num,
                    "maze": ex["prompt"],   # includes the maze text
                    "gold_start": gold,
                    "pred_start": pred,
                    "correct": is_correct,
                    "raw_output": output
                })

                # if wrong, keep a simplified view
                if not is_correct:
                    wrong_results.append({
                        "file": filename,
                        "example": line_num,
                        "maze": ex["prompt"],
                        "gold_start": gold,
                        "pred_start": pred,
                        "raw_output": output
                    })

        accuracy = correct / total if total > 0 else 0
        print(f"✅ {filename}: {correct}/{total} correct ({accuracy:.2%})")

        # summary row
        all_results.append({
            "file": filename,
            "example": "SUMMARY",
            "maze": None,
            "gold_start": None,
            "pred_start": None,
            "correct": f"{correct}/{total}",
            "raw_output": f"Accuracy={accuracy:.4f}"
        })

    # --- Save full results ---
    out_csv = "maze_inference_results.csv"
    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["file", "example", "maze", "gold_start", "pred_start", "correct", "raw_output"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n📊 All results saved to {out_csv}")

    # --- Save wrong answers separately ---
    wrong_csv = "maze_inference_wrong.csv"
    with open(wrong_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["file", "example", "maze", "gold_start", "pred_start", "raw_output"])
        writer.writeheader()
        writer.writerows(wrong_results)
    print(f"❌ Wrong answers saved to {wrong_csv}")


if __name__ == "__main__":
    main()
