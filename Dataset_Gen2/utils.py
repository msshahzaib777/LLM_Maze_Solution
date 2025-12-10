import json
import math
import os
from typing import List, Dict, Any, Optional, Tuple
import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

TASKS = ("OPTIMAL_NEXT_STEP",)  # Simplified to single task

DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

def suggest_optimal_max_seq_length(jsonl_path: str, percentile: float = 0.995, safety_margin: float = 1.05, *, tokenizer_override=None) -> Dict[str, float]:
    """Estimate a safe max sequence length for a JSONL dataset."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1].")

    tok = tokenizer_override or tokenizer
    _completion = lambda e: json.dumps(v) if isinstance(v := e.get("completion", e.get("target", "")), (dict, list)) else (v or "")

    with open(jsonl_path) as f:
        lengths = sorted([
            len(tok.encode(s.get("prompt", ""), add_special_tokens=True)) + 
            len(tok.encode(_completion(s), add_special_tokens=False))
            for s in map(json.loads, f)
        ])

    if not lengths:
        raise ValueError(f"No records found in {jsonl_path}")

    idx = min(len(lengths) - 1, math.ceil(percentile * len(lengths)) - 1)
    return {
        "recommended": max(lengths[-1], math.ceil(lengths[idx] * safety_margin)),
        "max_observed": lengths[-1],
        "percentile_length": lengths[idx],
        "percentile": percentile,
        "dataset_size": len(lengths),
    }

clean_maze_ascii = lambda maze: maze.replace("+", " ")

def find_start(maze_ascii: str):
    return next(((i, j) for i, row in enumerate(maze_ascii.splitlines()) for j, c in enumerate(row) if c == "S"), None)

def find_end(maze_ascii: str):
    return next(((i, j) for i, row in enumerate(maze_ascii.splitlines()) for j, c in enumerate(row) if c == "E"), None)

def maze_value_at(maze, position, default: str = "#"):
    lines = maze.splitlines()
    i, j = position
    if not (0 <= i < len(lines)):
        return default
    row = lines[i]
    if not (0 <= j < len(row)):
        return default
    return row[j]

def walkable_directions(maze_ascii: str, position: Tuple[int, int]) -> Dict[str, bool]:
    return {name: maze_value_at(maze_ascii, (position[0] + di, position[1] + dj)) != "#" 
            for name, (di, dj) in DIRECTION_VECTORS.items()}

def apply_direction(position: Tuple[int, int], direction: str) -> Tuple[int, int]:
    if (di_dj := DIRECTION_VECTORS.get(direction.lower())):
        return position[0] + di_dj[0], position[1] + di_dj[1]
    raise ValueError(f"Unknown direction: {direction}")

def update_maze(maze, position, new_value):
    lines = maze.splitlines()
    lines[position[0]] = lines[position[0]][:position[1]] + new_value + lines[position[0]][position[1]+1:]
    return "\n".join(lines)

def get_correct_direction_at(maze: str, position, maze_size):
    """Get the optimal next direction from the solved maze."""
    i, j = position
    for d, (di, dj) in DIRECTION_VECTORS.items():
        ni, nj = i + di, j + dj
        if 0 <= ni <= maze_size and 0 <= nj <= maze_size:
            if maze_value_at(maze, (ni, nj)) in ["+", "E"]:
                return d
    return None

def get_local_view(maze, current, visited, view_radius=2):
    """Get local view of surrounding cells in a grid around current position."""
    r, c = current
    return "\n".join(
        " ".join(
            "C" if (r+i, c+j) == current else 
            "-" if (r+i, c+j) in visited else 
            maze_value_at(maze, (r+i, c+j), "#")
            for j in range(-view_radius, view_radius + 1)
        )
        for i in range(-view_radius, view_radius + 1)
    )

def calculate_distances_to_walls(maze, current):
    """Calculate distances to walls in all 4 directions."""
    distances = {}
    for direction in ["up", "down", "left", "right"]:
        dist, pos = 0, current
        while maze_value_at(maze, (pos := apply_direction(pos, direction)), "#") != "#":
            dist += 1
        distances[direction] = dist
    return distances

def get_solution_with_env_interaction(maze_ascii: str, solved_maze: str, maze_size: int, max_visits: int = 10):
    """Generate training samples with environment-agent interaction."""
    maze, start, goal = maze_ascii, find_start(maze_ascii), find_end(maze_ascii)
    current, move_history, visit_counts, visited, samples = start, [], {start: 1}, {start}, []
    
    for step in range(maze_size * maze_size * 2):
        if current == goal or visit_counts.get(current, 0) > max_visits:
            break
        
        optimal_dir = get_correct_direction_at(solved_maze, current, maze_size)
        if not optimal_dir:
            break
        
        samples.append({
            "position": current, "goal": goal,
            "local_view": get_local_view(maze, current, visited, 2),
            "distances": calculate_distances_to_walls(maze, current),
            "history_str": " -> ".join(move_history[-5:]) if move_history else "None",
            "optimal_step": optimal_dir,
            "visit_count": visit_counts.get(current, 0),
            "step": step
        })
        
        # Move and update state
        visited.add(current)
        maze = update_maze(maze, current, "-")
        current = apply_direction(current, optimal_dir)
        move_history.append(optimal_dir)
        visit_counts[current] = visit_counts.get(current, 0) + 1
    
    return samples

def make_training_example(m, tasks: List[str] = ["OPTIMAL_NEXT_STEP"], id: Optional[str] = None) -> Dict[str, Any]:
    """Generate training example with environment-agent interaction approach."""
    maze_ascii = clean_maze_ascii(m.tostring(True, True))
    return {
        "id": id, "maze": maze_ascii, "start": m.start, "end": getattr(m, "end", None),
        "maze_size": m.generator.H,
        "solution_samples": get_solution_with_env_interaction(maze_ascii, str(m), m.generator.H),
        "solved_maze": str(m)
    }

def dict_to_prompt_completion(ex, tasks=None):
    """Convert maze example into training instances with environment-agent format."""
    maze_id, goal = ex.get("id", "unknown"), ex["end"]
    training_examples = []
    
    for idx, s in enumerate(ex.get("solution_samples", [])):
        d = s['distances']
        step_prompt = (
            f"You are navigating a maze. Analyze the environment and choose the optimal next move.\n\n"
            f"Current State:\n  Position: {s['position']}\n  Goal: {goal}\n  Visit count at current cell: {s['visit_count']}\n\n"
            f"Environment Info:\n  Distance to walls - Up: {d['up']}, Down: {d['down']}, Left: {d['left']}, Right: {d['right']}\n"
            f"  Recent moves: {s['history_str']}\n\n"
            f"Local View (5x5 grid):\nLegend: C=You, -=Visited, #=Wall, E=Goal, space=Open\n{s['local_view']}\n\n"
            f"Task: Choose optimal direction (up/down/left/right) to reach goal efficiently.\n"
            f"Avoid: Walls, recently visited cells, and backtracking.\n"
            f"Return: JSON format with optimal_step key"
        )
        
        # Build reasoning
        pos, opt = s['position'], s['optimal_step']
        rd, cd = goal[0] - pos[0], goal[1] - pos[1]
        avail = ', '.join([d for d in ['up', 'down', 'left', 'right'] if s['distances'][d] > 0])
        
        move_type = ("UP toward goal" if opt == 'up' and rd < 0 else
                    "DOWN toward goal" if opt == 'down' and rd > 0 else
                    "LEFT toward goal" if opt == 'left' and cd < 0 else
                    "RIGHT toward goal" if opt == 'right' and cd > 0 else
                    f"{opt.upper()} around obstacle")
        
        reasoning = f"Position: {pos}, Goal: {goal}. Need: {'down' if rd > 0 else 'up' if rd < 0 else 'same'} {abs(rd)} rows, " \
                   f"{'right' if cd > 0 else 'left' if cd < 0 else 'same'} {abs(cd)} cols. Available: {avail}. " \
                   f"History: {s['history_str']}. {move_type}" + \
                   (f" (visit {s['visit_count']})" if s['visit_count'] > 1 else "") + f". Optimal: {opt}"
        
        training_examples.append({
            "id": f"{maze_id}_step_{idx}", "task": "OPTIMAL_NEXT_STEP",
            "prompt": build_prompt(step_prompt),
            "completion": f"<think>{reasoning}</think>\n" + json.dumps({"optimal_step": opt}, ensure_ascii=False),
            "metadata": {"position": pos, "goal": goal, "step": s["step"], "visit_count": s["visit_count"], "distances": d}
        })
    
    return "\n".join(json.dumps(ex) for ex in training_examples) + "\n"

def build_prompt(user_prompt: str) -> str:
    """Build chat-formatted prompt."""
    return tokenizer.apply_chat_template([
        {"role": "system", "content": 
            "You are an expert maze navigator AI. Your task:\n"
            "1. Analyze the local 5x5 view around your current position\n"
            "2. Consider distances to walls and movement history\n"
            "3. Choose the optimal direction (up/down/left/right) to reach the goal\n"
            "4. Avoid walls, minimize revisiting cells, and progress toward the goal\n"
            "5. Think step-by-step in <think> tags, then return JSON: {\"optimal_step\": \"direction\"}"},
        {"role": "user", "content": user_prompt}
    ], tokenize=False, add_generation_prompt=True)

def save_jsonl(examples, path, mapper=None):
    with open(path, "w") as f:
        for ex in examples:
            f.write(mapper(ex))

def open_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
