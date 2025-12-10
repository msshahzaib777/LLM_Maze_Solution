import torch, os, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
from calculate_accuracy import calculate_accuracy
from transformers import BitsAndBytesConfig
from collections import Counter, defaultdict
from sklearn.utils import shuffle
import constant as config
from constant import MERGED_MODEL_DIR, ADAPTER_DIR, DATA_DIR        

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")
    device = torch.device("cuda:0")
    target_dtype = torch.float16
else:
    device = torch.device("cpu")
    target_dtype = torch.float32

# Global config
base_id, Dataset_NAME, ADAPTER_NAME, step_num, BATCH_SIZE, MAX_NEW_TOKENS = (
    config.base_id, config.Dataset_NAME, config.ADAPTER_NAME, config.step_num, config.test_BATCH_SIZE, config.INF_MAX_NEW_TOKENS)
adapter_base_path = f"{ADAPTER_DIR}/{ADAPTER_NAME}"

def find_best_checkpoint(adapter_base_path, manual_step=None):
    dirs = sorted([d for d in os.listdir(adapter_base_path) if d.startswith('step_')], key=lambda x: int(x.split('_')[1]))
    if not dirs: exit(1)
    if manual_step:
        return os.path.join(adapter_base_path, f"step_{manual_step}"), manual_step
    state_path = os.path.join(adapter_base_path, dirs[-1], "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location='cpu')
        eval_losses = state.get('eval_losses', [])
        if eval_losses:
            steps = [int(d.split('_')[1]) for d in dirs]
            best_step = min(steps, key=lambda x: abs(x - ((eval_losses.index(min(eval_losses)) + 1) * 50)))
            return os.path.join(adapter_base_path, f"step_{best_step}"), best_step
    last = dirs[-1]
    return os.path.join(adapter_base_path, last), int(last.split('_')[1])

def extract_maze_size(example_id):
        # Example: "maze_8x8_task_XYZ_123" -> extract 8x8
    m = re.search(r'maze_(\d+)x(\d+)', example_id)
    return f"{m.group(1)}x{m.group(2)}" if m else "UNKNOWN"
        
def filter_test_examples(all_examples, max_samples=3000):
    # Group by (task, maze_size)
    grouped = defaultdict(list)
    [grouped[(ex.get('task', 'UNKNOWN'), extract_maze_size(ex['id']))].append(ex) for ex in all_examples]
    per_group = max(1, max_samples // len(grouped))
    selected = [ex for group in grouped.values() for ex in shuffle(group, random_state=42)[:per_group]]
    if len(selected) < max_samples:
        rem = [ex for group in grouped.values() for ex in group if ex not in selected]
        selected += shuffle(rem, random_state=42)[:max_samples - len(selected)]
    return selected[:max_samples]

def eval_results(preds_jsonl):
    if not os.path.exists(preds_jsonl): return
    try:
        results, text_output = calculate_accuracy(preds_jsonl, verbose=True, save_text_report=True)
        out_dir = Path(preds_jsonl).parent
        with open(out_dir / "accuracy_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        if text_output:
            with open(out_dir / "accuracy_report.txt", 'w') as f:
                f.write('\n'.join(text_output))
        print(f"Overall Accuracy: {results['overall']['accuracy']:.2f}%")
    except Exception as e:
        print(f"Accuracy error: {e}")

def main():
    base_id, Dataset_NAME, ADAPTER_NAME, step_num, BATCH_SIZE, MAX_NEW_TOKENS = (
    config.base_id, config.Dataset_NAME, config.ADAPTER_NAME, config.step_num, config.test_BATCH_SIZE, config.INF_MAX_NEW_TOKENS)
    adapter_base_path = f"{ADAPTER_DIR}/{ADAPTER_NAME}"

    best_checkpoint, step_num = find_best_checkpoint(adapter_base_path, step_num)
    print(f"Using checkpoint: {best_checkpoint} (step {step_num})")

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if config.quantized:
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    else:
        quant_config = None
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=target_dtype,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    merged = torch.compile(PeftModel.from_pretrained(base, best_checkpoint, is_trainable=False).merge_and_unload(), mode="reduce-overhead")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference (DataParallel)")
        merged = torch.nn.DataParallel(merged)

    gen_config = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        do_sample=False,
        num_beams=1
    )

    eval_dir = os.path.join(DATA_DIR, Dataset_NAME, "eval", f"{ADAPTER_NAME}_step_{step_num}")
    os.makedirs(eval_dir, exist_ok=True)
    preds_jsonl = os.path.join(eval_dir, "test_predictions.jsonl")

    with open(f"{DATA_DIR}/{Dataset_NAME}/test.jsonl", 'r') as f:
        test_examples = filter_test_examples([json.loads(line) for line in f], max_samples=3000)
    if os.path.exists(preds_jsonl):
        with open(preds_jsonl, 'r') as f:
            existing = {json.loads(line)['id'] for line in f}
        test_examples = [ex for ex in test_examples if ex['id'] not in existing]
    if not test_examples:
        print("All examples processed"); return
    merged.eval(); processed = 0; batch_size = BATCH_SIZE if BATCH_SIZE > 0 else config.test_BATCH_SIZE
    with open(preds_jsonl, 'a') as outfile:
        pbar = tqdm(total=len(test_examples), desc="Inference")
        for i in range(0, len(test_examples), batch_size):
            batch = test_examples[i:i+batch_size]
            prompts = [ex['prompt'] for ex in batch]
            inputs = tok(prompts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                if isinstance(merged, torch.nn.DataParallel):
                    outputs = merged.module.generate(**inputs, **gen_config)
                else:
                    outputs = merged.generate(**inputs, **gen_config)
            for j, ex in enumerate(batch):
                gen_tokens = outputs[j, inputs['input_ids'].shape[1]:]
                generated = tok.decode(gen_tokens, skip_special_tokens=True).strip()
                if generated.startswith(ex['prompt']): generated = generated[len(ex['prompt']):].strip()
                outfile.write(json.dumps({
                    'id': ex.get('id', processed),
                    'prompt': ex['prompt'],
                    'target': ex.get('completion', ''),
                    'prediction': generated,
                    'task': ex.get('task', 'UNKNOWN')
                }) + '\n')
                processed += 1
            pbar.update(len(batch))
        outfile.flush()
    print(f"Processed: {processed}, Saved to: {preds_jsonl}"); eval_results(preds_jsonl)

if __name__ == "__main__":
    main()