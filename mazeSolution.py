import json
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constant import MERGED_MODEL_DIR, INF_MAX_NEW_TOKENS
import constant as config
from pytorchInference import find_best_checkpoint
from transformers import BitsAndBytesConfig
from peft import PeftModel
from Dataset_Gen.datasetGeneration import generate_single_maze
from Dataset_Gen.utils import maze_value_at, apply_direction, update_maze, walkable_directions, build_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def extract_direction(response):
    """Extract direction from model response (JSON or fallback to keywords)."""
    try:
        start, end = response.rfind("{"), response.rfind("}") + 1
        return json.loads(response[start:end]).get("optimal_step", "").lower()
    except:
        return next((d for d in ["up", "down", "left", "right"] if d in response.lower()), None)

def get_local_view(maze, current, visited, view_radius=2):
    """Get local view of surrounding cells in a grid around current position.
    
    Args:
        maze: The full maze string
        current: Current position (row, col)
        visited: Set/dict of visited positions
        view_radius: How many cells to show in each direction (default 2 for 5x5 grid)
    """
    lines = maze.splitlines()
    local_view = []
    surroundings = {"up": None, "down": None, "left": None, "right": None}
    
    curr_row, curr_col = current
    
    # Build a grid view centered on current position
    for i in range(curr_row - view_radius, curr_row + view_radius + 1):
        row_str = ""
        for j in range(curr_col - view_radius, curr_col + view_radius + 1):
            pos = (i, j)
            
            # Current position
            if pos == current:
                row_str += "C "
            # Get cell value
            else:
                cell = maze_value_at(maze, pos, "#")
                # Mark visited cells
                if pos in visited:
                    cell = "-"
                row_str += cell + " "
            
            # Track immediate neighbors for validation
            if i == curr_row - 1 and j == curr_col:
                surroundings["up"] = maze_value_at(maze, pos, "#")
            elif i == curr_row + 1 and j == curr_col:
                surroundings["down"] = maze_value_at(maze, pos, "#")
            elif i == curr_row and j == curr_col - 1:
                surroundings["left"] = maze_value_at(maze, pos, "#")
            elif i == curr_row and j == curr_col + 1:
                surroundings["right"] = maze_value_at(maze, pos, "#")
        
        local_view.append(row_str.rstrip())
    
    return "\n".join(local_view), surroundings

def calculate_distances_to_walls(maze, current):
    """Calculate distances to walls in all 4 directions."""
    distances = {}
    for direction in ["up", "down", "left", "right"]:
        dist = 0
        pos = current
        while True:
            try:
                pos = apply_direction(pos, direction)
                cell = maze_value_at(maze, pos, "#")
                if cell == "#":
                    break
                dist += 1
            except:
                break
        distances[direction] = dist
    return distances

def solve_maze(model, tokenizer, maze_size=5, max_steps=50, max_illegal=5):
    """Solve maze step-by-step using trained model with local view and visit constraints."""
    # Generate maze
    print(f"Generating {maze_size}x{maze_size} maze...")
    maze_ex = generate_single_maze((maze_size, 0, {"repeat": 3, "entrances": 5, "difficulty": 0.8}, False))
    maze = maze_ex["maze"]
    current = tuple(maze_ex["answer"]["start"])
    goal = tuple(maze_ex["answer"]["end"])
    
    # Track state
    path = []
    illegal_count = 0
    visited = {current: 1}  # Track visit count per cell
    move_history = []
    max_visits_per_cell = 10
    global_move_budget = maze_size * maze_size * 2  # Generous budget
    
    print(f"\nStart: {current}, Goal: {goal}\n{maze}\n{'='*50}")
    
    # Solve step by step
    for step in range(1, max_steps + 1):
        # Check if reached goal
        if current == goal or maze_value_at(maze, current) == "E":
            break
        
        # Check global move budget
        if step > global_move_budget:
            print(f"ERROR: Exceeded global move budget ({global_move_budget})")
            return {"solved": False, "steps": step, "path": path, "failure_reason": "global_budget_exceeded", "illegal_moves": illegal_count}
            
        # Get local view and environmental info
        local_view, surroundings = get_local_view(maze, current, visited)
        distances = calculate_distances_to_walls(maze, current)
        
        # Build prompt with environmental information
        history_str = " -> ".join(move_history[-5:]) if move_history else "None"
        prompt = build_prompt(
            f"Current position: {current}\n"
            f"Goal position: {goal}\n"
            f"Distances to walls: up={distances['up']}, down={distances['down']}, left={distances['left']}, right={distances['right']}\n"
            f"Recent history: {history_str}\n"
            f"Cell visit count: {visited.get(current, 0)}\n\n"
            f"Local view (C=current, -=visited, #=wall, E=exit):\n{local_view}\n\n"
            f"Return JSON with optimal_step (direction: up/down/left/right). Minimize reasoning."
        )
        prompt = prompt + "<think>"
        # Log the input prompt
        print(f"\n--- Step {step} Input ---")
        print(prompt)
        print("--- End Input ---\n")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=INF_MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        
        direction = extract_direction(response)
        print(f"Step {step}: {current} -> {direction} | Visits: {visited.get(current, 0)}")
        
        # Validate direction extraction
        if not direction:
            illegal_count += 1
            print(f"ERROR: No direction extracted (illegal {illegal_count}/{max_illegal})")
            if illegal_count >= max_illegal:
                return {"solved": False, "steps": step, "path": path, "failure_reason": "no_direction_extracted", "illegal_moves": illegal_count}
            continue
        
        # Validate move is legal
        if surroundings.get(direction, "#") == "#":
            illegal_count += 1
            print(f"ERROR: Invalid move '{direction}' into wall (illegal {illegal_count}/{max_illegal})")
            if illegal_count >= max_illegal:
                return {"solved": False, "steps": step, "path": path, "failure_reason": "invalid_moves", "illegal_moves": illegal_count}
            continue
        
        # Move to next position
        next_pos = apply_direction(current, direction)
        
        # Check visit constraint
        if visited.get(next_pos, 0) >= max_visits_per_cell:
            print(f"WARNING: Cell {next_pos} already visited {visited[next_pos]} times (max: {max_visits_per_cell})")
            # Allow the move but it will likely lead to failure
        
        # Update state
        current = next_pos
        visited[current] = visited.get(current, 0) + 1
        path.append(direction)
        move_history.append(direction)
        
        # Check for looping (same cell visited too many times)
        if visited[current] > max_visits_per_cell:
            print(f"ERROR: Cell {current} visited {visited[current]} times (exceeded max {max_visits_per_cell})")
            return {"solved": False, "steps": step, "path": path, "failure_reason": "visit_limit_exceeded", "illegal_moves": illegal_count}
    
    solved = current == goal or maze_value_at(maze, current) == "E"
    failure_reason = None if solved else ("max_steps_reached" if step >= max_steps else "unknown")
    
    # Mark path on maze for visualization
    display_maze = maze
    for pos, count in visited.items():
        if pos != tuple(maze_ex["answer"]["start"]) and pos != goal:
            display_maze = update_maze(display_maze, pos, "-")
    
    print(f"\n{'='*50}\n{'SOLVED' if solved else 'FAILED'} in {len(path)} steps")
    print(f"Path: {' -> '.join(path)}")
    print(f"Max visits to any cell: {max(visited.values())}")
    print(f"\n{display_maze}")
    
    return {"solved": solved, "steps": len(path), "path": path, "failure_reason": failure_reason, "illegal_moves": illegal_count, "max_visits": max(visited.values())}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MERGED_MODEL_DIR)
    parser.add_argument("--maze_size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_mazes", type=int, default=1, help="Number of mazes to test")
    args = parser.parse_args()
    
    # Setup logging to file with streaming
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/maze_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = open(log_file, 'w', buffering=1)
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    sys.stdout = Tee(sys.stdout, log)
    print(f"Logging to: {log_file}")
    
    target_dtype = torch.float16

    print(f"Loading model from {args.model_path}...")
    base_id, Dataset_NAME, ADAPTER_NAME, step_num, BATCH_SIZE, INF_MAX_NEW_TOKENS = (
    config.base_id, config.Dataset_NAME, config.ADAPTER_NAME, config.step_num, config.test_BATCH_SIZE, config.INF_MAX_NEW_TOKENS)
    
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if config.quantized:
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    else:
        quant_config = None
    
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=target_dtype,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    
    # Handle adapter loading
    if ADAPTER_NAME is not None:
        adapter_base_path = f"{config.ADAPTER_DIR}/{ADAPTER_NAME}"
        best_checkpoint, step_num = find_best_checkpoint(adapter_base_path, step_num)
        print(f"Using checkpoint: {best_checkpoint} (step {step_num})")
        model = torch.compile(PeftModel.from_pretrained(base, best_checkpoint, is_trainable=False).merge_and_unload(), mode="reduce-overhead")
    else:
        print("No adapter specified, using base model only")
        model = torch.compile(base, mode="reduce-overhead")
    gen_config = dict(
        max_new_tokens=INF_MAX_NEW_TOKENS,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        do_sample=False,
        num_beams=1
    )
    tokenizer = tok
    # Run multiple mazes and track results
    results = []
    for i in range(args.num_mazes):
        if args.num_mazes > 1:
            print(f"\n{'='*50}\nMaze {i+1}/{args.num_mazes}\n{'='*50}")
        result = solve_maze(model, tokenizer, args.maze_size)
        results.append(result)
    
    # Summary statistics
    if args.num_mazes > 1:
        solved = sum(r["solved"] for r in results)
        success_rate = solved / args.num_mazes * 100
        avg_steps = sum(r["steps"] for r in results) / args.num_mazes
        avg_max_visits = sum(r.get("max_visits", 1) for r in results) / args.num_mazes
        failures = {}
        for r in results:
            if r.get("failure_reason"):
                failures[r["failure_reason"]] = failures.get(r["failure_reason"], 0) + 1
        
        print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
        print(f"Success Rate: {solved}/{args.num_mazes} ({success_rate:.1f}%)")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Max Visits per Cell: {avg_max_visits:.1f}")
        if failures:
            print("Failure Reasons:")
            for reason, count in failures.items():
                pct = count / args.num_mazes * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")
    
    log.close()
    print(f"\nLog saved to: {log_file}")
