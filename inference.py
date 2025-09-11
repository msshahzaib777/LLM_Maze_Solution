"""
Inference examples for MLX model: mlx-community/Qwen3-235B-A22B-8bit

Install deps:
  pip install -U mlx mlx-lm

Quick CLI (no Python needed):
  mlx_lm.generate \
    --model mlx-community/Qwen3-235B-A22B-8bit \
    --eos-token "<|endoftext|>" \
    --prompt "Explain Mixture-of-Experts in simple terms." \
    --max-tokens 256

This script shows:
  1) Plain text prompting
  2) Chat-style prompting (system + user) with a safe fallback template

Note:
  • This is a *huge* MoE model. Ensure you have ample unified memory.
  • For Qwen-family models, setting eos to "<|endoftext|>" is usually correct.
"""

import argparse
import sys

import mlx as mx  # MLX core (unified memory on Apple Silicon)
from mlx_lm import load, generate


# ---------------------------
# Utilities
# ---------------------------

def ensure_eos(tokenizer, eos_token: str | None):
    """Ensure tokenizer has a valid eos token, optionally overriding.
    Returns the possibly modified tokenizer.
    """
    if eos_token:
        tokenizer.eos_token = eos_token
        if getattr(tokenizer, "eos_token_id", None) is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            except Exception:
                pass
    else:
        # If eos is missing entirely, default to Qwen-style
        if getattr(tokenizer, "eos_token", None) in (None, ""):
            tokenizer.eos_token = "<|endoftext|>"
    # Pad-as-eos is fine for generation-only
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "pad_token_id") and getattr(tokenizer, "pad_token_id", None) is None:
            try:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            except Exception:
                pass
    return tokenizer


def apply_chat_template_safe(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Use tokenizer.apply_chat_template if available, otherwise fall back to a simple Qwen-like template."""
    tpl = getattr(tokenizer, "apply_chat_template", None)
    if callable(tpl):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    # Fallback: Qwen-style minimal template
    # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------
# Inference paths
# ---------------------------

def infer_prompt(model, tokenizer, prompt: str, max_tokens: int):
    return generate(model, tokenizer, prompt, max_tokens=max_tokens)


def infer_chat(model, tokenizer, system_msg: str, user_msg: str, max_tokens: int):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    formatted = apply_chat_template_safe(tokenizer, messages, add_generation_prompt=True)
    return generate(model, tokenizer, formatted, max_tokens=max_tokens)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference for Qwen3-235B-A22B-8bit on MLX")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-235B-A22B-8bit")
    parser.add_argument("--mode", choices=["prompt", "chat"], default="prompt")
    parser.add_argument("--prompt", type=str, default="Explain Mixture-of-Experts in simple terms.")
    parser.add_argument("--system", type=str, default="You are a concise, helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Optional: set RNG seed (affects sampling in some configs)
    try:
        mx.random.seed(args.seed)
    except Exception:
        pass

    print(f"Loading model: {args.model}", flush=True)
    model, tokenizer = load(args.model)
    tokenizer = ensure_eos(tokenizer, args.eos_token)

    if args.mode == "prompt":
        out = infer_prompt(model, tokenizer, args.prompt, args.max_tokens)
    else:  # chat
        out = infer_chat(model, tokenizer, args.system, args.prompt, args.max_tokens)

    print("\n=== Output ===")
    print(out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
