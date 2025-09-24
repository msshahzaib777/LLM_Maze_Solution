from Dataset_Gen.utils import build_prompt
from mlx_lm import generate, load

# --------------------------
# Load with adapter & test
# --------------------------
model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
adapter_dir = "finetuned_model/adapters"
model_lora, tokenizer = load(model_path, adapter_path=adapter_dir)

# Tiny sanity check on a single sample prompt
demo_maze = (
    "###########\n"
    "# # # # # #\n"
    "###########\n"
    "# # # # # #\n"
    "###########\n"
    "#     #  E#\n"
    "# ### ### #\n"
    "# # #   # #\n"
    "# ####### #\n"
    "#        S#\n"
    "###########"
)
demo_prompt = "Identify the start location and its available directions in this maze."
demo_infer = build_prompt(demo_maze, demo_prompt)
resp = generate(model_lora, tokenizer, prompt=demo_infer, max_tokens=64, verbose=True)
print("\nMODEL OUTPUT:\n", resp)
