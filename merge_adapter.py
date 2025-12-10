

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel
import argparse, os

def merge_adapter(base_model_name, adapter_path, output_path):
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    quant_config = None
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    merged = torch.compile(
        PeftModel.from_pretrained(base, adapter_path, is_trainable=False).merge_and_unload(),
        mode="reduce-overhead"
    )
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter_path, args.output_path)