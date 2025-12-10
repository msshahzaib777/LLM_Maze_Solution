import json
import math
import os
from typing import List, Dict, Any, Optional, Tuple
import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

TASKS = ("DETECT_START_END", "AVAILABLE_DIRECTIONS", "VALID_MOVE", "OPTIMAL_NEXT_STEP", "MAZE_SOLUTION")

DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

DIR_TEXT_TO_ENUM = {
    "up": "U",
    "down": "D",
    "left": "L",
    "right": "R",
}

DIR_ENUM_TO_TEXT = {v: k for k, v in DIR_TEXT_TO_ENUM.items()}

def suggest_optimal_max_seq_length(
    jsonl_path: str,
    percentile: float = 0.995,
    safety_margin: float = 1.05,
    *,
    tokenizer_override=None
) -> Dict[str, float]:
    """
    Estimate a safe max sequence length for a JSONL dataset and return summary stats.

    Args:
        jsonl_path: Path to the JSONL file containing `prompt` plus `completion` or `target`.
        percentile: Fraction of samples that should fit without truncation.
        safety_margin: Multiplier applied to the percentile length.
        tokenizer_override: Optional tokenizer to use instead of the module-level tokenizer.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1].")

    tok = tokenizer_override or tokenizer

    def _completion(entry: Dict[str, Any]) -> str:
        value = entry.get("completion", entry.get("target", ""))
        return json.dumps(value) if isinstance(value, (dict, list)) else (value or "")

    with open(jsonl_path, "r") as handle:
        lengths = [
            len(tok.encode(sample.get("prompt", ""), add_special_tokens=True))
            + len(tok.encode(_completion(sample), add_special_tokens=False))
            for sample in map(json.loads, handle)
        ]

    if not lengths:
        raise ValueError(f"No records found in {jsonl_path}")

    lengths.sort()
    idx = min(len(lengths) - 1, math.ceil(percentile * len(lengths)) - 1)
    percentile_len = lengths[idx]
    max_len = lengths[-1]
    recommended = max(max_len, math.ceil(percentile_len * safety_margin))

    return {
        "recommended": recommended,
        "max_observed": max_len,
        "percentile_length": percentile_len,
        "percentile": percentile,
        "dataset_size": len(lengths),
    }


def clean_maze_ascii(maze_ascii: str) -> str:
    return maze_ascii.replace("+", " ")

def find_start(maze_ascii: str):
    lines = maze_ascii.splitlines()
    for i, row in enumerate(lines):
        for j, c in enumerate(row):
            if c == "S":
                return (i, j)
    return None

def find_end(maze_ascii: str):
    lines = maze_ascii.splitlines()
    for i, row in enumerate(lines):
        for j, c in enumerate(row):
            if c == "E":
                return (i, j)
    return None

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
    i, j = position
    lines = maze_ascii.splitlines()
    height = len(lines)
    result: Dict[str, bool] = {}
    for name, (di, dj) in DIRECTION_VECTORS.items():
        ni, nj = i + di, j + dj
        in_bounds = 0 <= ni < height
        if in_bounds:
            row = lines[ni]
            in_bounds = 0 <= nj < len(row)
        result[name] = in_bounds and maze_value_at(maze_ascii, (ni, nj)) != "#"
    return result

def apply_direction(position: Tuple[int, int], direction: str) -> Tuple[int, int]:
    key = direction.lower()
    if key in DIRECTION_VECTORS:
        di, dj = DIRECTION_VECTORS[key]
    else:
        enum_key = direction.upper()
        if enum_key not in DIR_ENUM_TO_TEXT:
            raise ValueError(f"Unknown direction: {direction}")
        di, dj = DIRECTION_VECTORS[DIR_ENUM_TO_TEXT[enum_key]]
    return position[0] + di, position[1] + dj

def get_surroundings(maze_ascii: str, start):
    directions = walkable_directions(maze_ascii, start)
    return directions

def get_random_points(maze_ascii: str, n: int = 5) -> List[Tuple[int, int]]:
    """Get n random walkable points from the maze."""
    lines = maze_ascii.splitlines()
    points = []
    height = len(lines)
    width = len(lines[0])
    
    while len(points) < n:
        i = random.randint(0, height-1)
        j = random.randint(0, width-1)
        if maze_value_at(maze_ascii, (i,j)) in [" ", "S", "E"]:
            points.append((i,j))
    return points

def get_directions_with_reasoning(maze_ascii: str, n_points: int = 5) -> List[Dict[str, Any]]:
    """Get walkable directions with detailed reasoning for n random points in the maze."""
    points = get_random_points(maze_ascii, n_points)
    results = []
    lines = maze_ascii.splitlines()
    
    for point in points:
        i, j = point
        walkable = walkable_directions(maze_ascii, point)
        current_row = lines[i]
        above_row = lines[i-1] if i > 0 else None 
        below_row = lines[i+1] if i < len(lines)-1 else None
        
        reasoning = f"At ({i},{j}). "
        
        # Check up
        if walkable.get("up"):
            reasoning += f"Up: '{maze_value_at(maze_ascii, (i-1,j))}' walkable. "
        else:
            cell = "#" if i == 0 else maze_value_at(maze_ascii, (i-1,j))
            reasoning += f"Up: '{cell}' blocked. "
            
        # Check down
        if walkable.get("down"):
            reasoning += f"Down: '{maze_value_at(maze_ascii, (i+1,j))}' walkable. "
        else:
            cell = "#" if i == len(lines)-1 else maze_value_at(maze_ascii, (i+1,j))
            reasoning += f"Down: '{cell}' blocked. "
            
        # Check left
        if walkable.get("left"):
            reasoning += f"Left: '{maze_value_at(maze_ascii, (i,j-1))}' walkable. "
        else:
            cell = "#" if j == 0 else maze_value_at(maze_ascii, (i,j-1))
            reasoning += f"Left: '{cell}' blocked. "
            
        # Check right
        if walkable.get("right"):
            reasoning += f"Right: '{maze_value_at(maze_ascii, (i,j+1))}' walkable."
        else:
            cell = "#" if j == len(current_row)-1 else maze_value_at(maze_ascii, (i,j+1))
            reasoning += f"Right: '{cell}' blocked."
        
        results.append({
            "position": point,
            "walkable_directions": [k for k,v in walkable.items() if v],
            "reasoning": reasoning
        })
        
    return results

def generate_move_samples(maze_ascii: str, n_points: int = 5) -> List[Dict[str, Any]]:
    """Generate positive and negative move samples from random points."""
    points = get_random_points(maze_ascii, n_points)
    samples = []
    lines = maze_ascii.splitlines()
    
    for point in points:
        i, j = point
        surroundings = walkable_directions(maze_ascii, point)
        # Generate samples for each direction
        for direction, is_valid in surroundings.items():
            di, dj = DIRECTION_VECTORS[direction]
            ni, nj = i + di, j + dj
            
            # Build reasoning based on direction
            if direction in ["up", "down"]:
                target_row = lines[ni] if 0 <= ni < len(lines) else None
                if target_row:
                    reasoning = f"From ({i},{j}) {direction}: '{target_row[j]}' - "
                else:
                    reasoning = f"From ({i},{j}) {direction}: out of bounds - "
                reasoning += "valid" if is_valid else "blocked"
            else:  # left or right
                if 0 <= nj < len(lines[i]):
                    reasoning = f"From ({i},{j}) {direction}: '{lines[i][nj]}' - "
                else:
                    reasoning = f"From ({i},{j}) {direction}: out of bounds - "
                reasoning += "valid" if is_valid else "blocked"
                
            samples.append({
                "position": point,
                "move": direction,
                "is_valid": is_valid,
                "reasoning": reasoning
            })
    return samples

def get_correct_direction_at(maze: str, position, maze_size):
    i, j = position
    for d, (di, dj) in DIRECTION_VECTORS.items():
        ni, nj = i + di, j + dj
        if 0 <= ni <= maze_size and 0 <= nj <= maze_size:
            if maze_value_at(maze, (ni, nj)) in ["+", "E"]:
                return d, (di, dj)

def update_maze(maze, position, new_value):
    lines = maze.splitlines()
    new_line = list(lines[position[0]])
    new_line[position[1]] = new_value
    lines[position[0]] = "".join(new_line)
    return "\n".join(lines)

def get_solution(maze_ascii: str, maze_size: int):
    solved_maze = maze_ascii
    maze = maze_ascii.replace("+", " ")
    start = find_start(maze)
    current = start
    stages = []
    
    while True:
        # === PROMPT/COMPLETION/THINK: SOLUTION ===
        # Exit if we reached the end
        if maze_value_at(solved_maze, current) == "E":
            break
            
        # Get next optimal direction
        direction = get_correct_direction_at(solved_maze, current, maze_size)
        if not direction:
            break
            
        # Calculate next position
        next_pos = apply_direction(current, direction[0])
        
        # Update both mazes - mark current position with '-'
        maze = update_maze(maze, current, "-")
        solved_maze = update_maze(solved_maze, current, "-")

        # Build reasoning for this stage
        reasoning = f"At ({current[0]},{current[1]}) -> {direction[0]} to ({next_pos[0]},{next_pos[1]})"
            
        # Store current stage info
        stage = {
            "position": current,
            "optimal_step": direction[0],
            "maze_before": maze,
            "maze_after": maze,
            "path": direction[0],
            "reasoning": reasoning
        }
        stages.append(stage)
        
        # Move to next position
        current = next_pos
    
    return stages

def generate_chain_of_thought(tasks: List[str], maze_ascii: str, start: Optional[Tuple[int, int]] = None,
                            end: Optional[Tuple[int, int]] = None,
                            surroundings: Optional[Dict[str, bool]] = None) -> Dict[str, str]:
    reasoning = {}

    for task in tasks:
        if task == "DETECT_START_END":
            lines = maze_ascii.splitlines()
            reasoning_text = ""
            if start is not None:
                s_row, s_col = start
                reasoning_text += f"S at ({s_row},{s_col}). "
            if end is not None:
                e_row, e_col = end
                reasoning_text += f"E at ({e_row},{e_col})."
            reasoning["start_end"] = reasoning_text
    return reasoning

def make_training_example(m, tasks: List[str] = ["DETECT_START_END"], id: Optional[str] = None) -> Dict[str, Any]:
    solved_maze = str(m)
    maze_ascii = clean_maze_ascii(m.tostring(True, True))
    start = m.start
    end = getattr(m, "end", None)
    surroundings = get_surroundings(maze_ascii, start)
    maze_size = m.generator.H
    solution = get_solution(solved_maze, maze_size)
    # Reduce n_points based on maze size for faster generation
    n_points = max(3, maze_size)  # Reduced from max(5, maze_size * 2)
    move_samples = generate_move_samples(maze_ascii, n_points=n_points)
    direction_samples = get_directions_with_reasoning(maze_ascii, n_points=n_points)
    return {
        "id": id,
        "maze": maze_ascii,
        "prompt": "Identify the start location which is labelled with S in this maze.",
        "chain_of_thought": generate_chain_of_thought(
            tasks=tasks,
            maze_ascii=maze_ascii, 
            start=start,
            end=end,
            surroundings=surroundings,
        ),
        "answer": {
            "start": start,
            "end": end,
        },
        "stages": solution,  # Includes stages with reasoning
        "solved_maze": solved_maze,
        "maze_size": maze_size,
        "move_samples": move_samples,
        "direction_samples": direction_samples
    }

def dict_to_prompt_completion(ex, tasks = None):
    """Convert a single maze example into multiple training instances for different tasks."""
    maze = ex["maze"]
    maze_id = ex.get("id", "unknown")
    chain_of_thought = ex.get("chain_of_thought", {})
    training_examples = []

    # === PROMPT/COMPLETION/THINK: DETECT_START_END ===
    start = ex["answer"]["start"]
    end = ex["answer"]["end"]
    start_end_prompt = (
        "Find coordinates of 'S' (start) and 'E' (end) in ASCII maze. Return JSON with keys: start [row,col], end [row,col].  Minimize reasoning and thinking.\n\n"
        f"<maze>\n{maze}\n</maze>"
    )
    completion = f"<think>{chain_of_thought.get('start_end', '')}</think>" + json.dumps({
        "start": start, 
        "end": end
    }, ensure_ascii=False)
    
    training_examples.append({
        "id": f"{maze_id}_start_end",
        "task": "DETECT_START_END",
        "prompt":  build_prompt(start_end_prompt),
        "completion": completion
    })

    # === PROMPT/COMPLETION/THINK: AVAILABLE_DIRECTIONS ===
    direction_samples = ex.get("direction_samples", [])
    for idx, sample in enumerate(direction_samples):
        position = sample["position"]
        directions = sample["walkable_directions"]
        dir_prompt = (
            f"From position {position}, list walkable directions. Return JSON with keys: available_directions (array).  Minimize reasoning and thinking.\n\n"
            f"<maze>\n{maze}\n</maze>"
        )
        completion = f"<think>{sample.get('reasoning', '')}</think>" + json.dumps({
            "available_directions": directions,
        }, ensure_ascii=False)
        
        training_examples.append({
            "id": f"{maze_id}_directions_{idx}",
            "task": "AVAILABLE_DIRECTIONS",
            "prompt":  build_prompt(dir_prompt),
            "completion": completion
        })

    # === PROMPT/COMPLETION/THINK: VALID_MOVE ===
    move_samples = ex.get("move_samples", [])
    for idx, sample in enumerate(move_samples):
        move_prompt = (
            f"From {sample['position']}, is moving {sample['move']} valid? Return JSON with keys: is_valid (boolean).  Minimize reasoning and thinking.\n\n"
            f"<maze>\n{maze}\n</maze>"
        )
        completion = f"<think>{sample.get('reasoning', '')}</think>" + json.dumps({
            "is_valid": sample["is_valid"],
        }, ensure_ascii=False)
        
        training_examples.append({
            "id": f"{maze_id}_valid_move_{idx}",
            "task": "VALID_MOVE", 
            "prompt":  build_prompt(move_prompt),
            "completion": completion
        })

    # === PROMPT/COMPLETION/THINK: OPTIMAL_NEXT_STEP ===
    solution_stages = ex.get("stages", [])
    for idx, stage in enumerate(solution_stages):
        step_prompt = (
            f"From {stage['position']}, what's the optimal next step to reach goal? Return JSON with keys: optimal_step (direction).  Minimize reasoning and thinking.\n\n"
            f"<maze>\n{maze}\n</maze>"
        )
        completion = f"<think>{stage.get('reasoning', '')}</think>" + json.dumps({
            "optimal_step": stage["optimal_step"],
        }, ensure_ascii=False)
        
        training_examples.append({
            "id": f"{maze_id}_optimal_step_{idx}",
            "task": "OPTIMAL_NEXT_STEP",
            "prompt": build_prompt(step_prompt),
            "completion": completion
        })
    # === PROMPT/COMPLETION/THINK: MAZE_SOLUTION ===
    if solution_stages:
        # Build thinking block
        full_solution_thinking = chain_of_thought.get('start_end', '') + " Path: "
        path_sequence = []
        
        for stage in solution_stages:
            full_solution_thinking += stage.get('reasoning', '') + ". "
            path_sequence.append(stage["optimal_step"])
            
        solution_prompt = (
            "Solve maze from 'S' to 'E'. Return JSON with keys: path (array of directions), think.\n\n"
            f"<maze>\n{maze}\n</maze>"
        )
        
        completion = f"<think>{full_solution_thinking}</think>" + json.dumps({
            "path": path_sequence,
        }, ensure_ascii=False)
        
        training_examples.append({
            "id": f"{maze_id}_full_solution",
            "task": "MAZE_SOLUTION",
            "prompt": build_prompt(solution_prompt),
            "completion": completion
        })
    # Convert each example to JSONL format
    return "\n".join(json.dumps(example) for example in training_examples) + "\n"

# --------------------------
# Dataset: JSONL → supervised pairs
# --------------------------
def build_prompt(user_prompt: str) -> str:
    # Keep formatting consistent so tokenization is stable.
    # === PROMPT/COMPLETION/THINK: system/user roles ===
    messages = [
        {"role": "system", "content": "Maze assistant. Respond in strict JSON with requested fields only. Minimize reasoning and thinking."},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_target(answer_start: List[int], answer_dirs: List[str]) -> str:
    # The trainer will learn only from these target tokens (prompt is masked out).
    return json.dumps({
        "start": [int(answer_start[0]), int(answer_start[1])],
        "available_directions": list(answer_dirs)
    }, ensure_ascii=False)

def save_jsonl(examples, path, mapper=None):
    with open(path, "w") as fout:
        for ex in examples:
            fout.write(mapper(ex))

def open_jsonl(input_jsonl_path):
    examples = []
    with open(input_jsonl_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples