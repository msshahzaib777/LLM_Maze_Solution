# inference_maze_fused.py
import json

from mlx_lm import load, generate
from Dataset_Gen.utils import make_training_example, dict_to_prompt_completion
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver

def main():
    # --- Path to fused model (output from mlx_lm.fuse) ---
    # fused_model_path = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"
    fused_model_path = "/Users/studentone/Documents/LLM_distributed Training/Fine-tuned_models/models/qwen3_12_32_10"   # update with your actual folder

    # --- Load fused model + tokenizer ---
    model, tokenizer = load(fused_model_path)
    g = 3
    m = Maze()
    m.generator = Prims(g, g)
    m.solver = BacktrackingSolver()
    m.generate_monte_carlo(20, 10, 0.5)
    examples = make_training_example(m)
    json_ex = dict_to_prompt_completion(examples, True, False, False)
    ex_prompt = json.loads(json_ex)
    # --- Generate answer ---
    output = generate(
        model,
        tokenizer,
        prompt=ex_prompt["prompt"],
        max_tokens=1000
    )
    print("Prompt:\n", ex_prompt["prompt"])
    print("\nModel Output:\n", output)

if __name__ == "__main__":
    main()
