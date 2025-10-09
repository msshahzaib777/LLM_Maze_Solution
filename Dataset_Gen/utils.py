import json
import math
import os
from typing import List, Dict, Any, Optional, Tuple
import random
from mlx_lm import load, generate

_, tokenizer = load("nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx")

TASKS = ("DETECT_START_END", "AVAILABLE_DIRECTIONS", "VALID_MOVE", "OPTIMAL_NEXT_STEP")

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

def is_walkable(maze_ascii: str, position: Tuple[int, int]) -> bool:
    return maze_value_at(maze_ascii, position) != "#"

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

def available_direction_enums(maze_ascii: str, position: Tuple[int, int]) -> List[str]:
    return [
        DIR_TEXT_TO_ENUM[name]
        for name, open_path in walkable_directions(maze_ascii, position).items()
        if open_path
    ]

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

def map_text_dirs_to_enum(text_dirs: List[str]) -> List[str]:
    out: List[str] = []
    for d in text_dirs:
        key = d.strip().lower()
        if key in DIR_TEXT_TO_ENUM:
            out.append(DIR_TEXT_TO_ENUM[key])
    return out

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
        
        reasoning = f"Analysis for point at (row={i}, col={j}):\n"
        reasoning += f"Current row: '{current_row}'\n"
        if above_row:
            reasoning += f"Row above:  '{above_row}'\n"
        if below_row:
            reasoning += f"Row below:  '{below_row}'\n"
        
        reasoning += "\nDirectional analysis:\n"
        # Check up
        if "up" in walkable:
            reasoning += f"UP: Can move up because cell above ({i-1},{j}) contains '{maze_value_at(maze_ascii, (i-1,j))}'\n"
        else:
            cell = "#" if i == 0 else maze_value_at(maze_ascii, (i-1,j))
            reasoning += f"UP: Cannot move up because cell above ({i-1},{j}) contains '{cell}' or is out of bounds\n"
            
        # Check down
        if "down" in walkable:
            reasoning += f"DOWN: Can move down because cell below ({i+1},{j}) contains '{maze_value_at(maze_ascii, (i+1,j))}'\n"
        else:
            cell = "#" if i == len(lines)-1 else maze_value_at(maze_ascii, (i+1,j))
            reasoning += f"DOWN: Cannot move down because cell below ({i+1},{j}) contains '{cell}' or is out of bounds\n"
            
        # Check left
        if "left" in walkable:
            reasoning += f"LEFT: Can move left because cell to left ({i},{j-1}) contains '{maze_value_at(maze_ascii, (i,j-1))}'\n"
        else:
            cell = "#" if j == 0 else maze_value_at(maze_ascii, (i,j-1))
            reasoning += f"LEFT: Cannot move left because cell to left ({i},{j-1}) contains '{cell}' or is out of bounds\n"
            
        # Check right
        if "right" in walkable:
            reasoning += f"RIGHT: Can move right because cell to right ({i},{j+1}) contains '{maze_value_at(maze_ascii, (i,j+1))}'\n"
        else:
            cell = "#" if j == len(current_row)-1 else maze_value_at(maze_ascii, (i,j+1))
            reasoning += f"RIGHT: Cannot move right because cell to right ({i},{j+1}) contains '{cell}' or is out of bounds\n"
        
        results.append({
            "position": point,
            "walkable_directions": {k: v for k,v in walkable.items() if v},
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
                current_row = lines[i]
                target_row = lines[ni] if 0 <= ni < len(lines) else None
                reasoning = (
                    f"Starting at position (row={i}, col={j}).\n"
                    f"Current row: '{current_row}'\n"
                )
                if target_row:
                    reasoning += f"Target row ({direction}): '{target_row}'\n"
                    reasoning += f"Character at target position is '{target_row[j]}': "
                else:
                    reasoning += f"Target row is out of bounds: "
                reasoning += "move is valid\n" if is_valid else "move is blocked\n"
            else:  # left or right
                current_row = lines[i]
                reasoning = (
                    f"Starting at position (row={i}, col={j}).\n"
                    f"Current row: '{current_row}'\n"
                    f"Moving {direction}, checking position {nj}: "
                )
                if 0 <= nj < len(current_row):
                    reasoning += f"character is '{current_row[nj]}': "
                else:
                    reasoning += "position is out of bounds: "
                reasoning += "move is valid\n" if is_valid else "move is blocked\n"
                
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
        reasoning = (
            f"At position (row={current[0]}, col={current[1]}).\n"
            f"Checking possible directions...\n"
            f"Optimal direction is '{direction[0]}' to move towards the goal.\n" 
            f"Moving {direction[0]} to position (row={next_pos[0]}, col={next_pos[1]}) by vector {direction[1]}.\n"
            f"Marking current position with '-' to track the path.\n"
            f"Maze after move:\n{maze}"
        )
            
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
                s_line = lines[s_row]
                reasoning_text += (
                    f"I scan the maze for the symbol 'S'.\n"
                    f"In row {s_row}, the line is: '{s_line}'\n"
                    f"'S' is at row={s_row}, col={s_col} (the {s_col}th character in the row).\n"
                )
            if end is not None:
                e_row, e_col = end
                e_line = lines[e_row]
                reasoning_text += (
                    f"Next, I look for the symbol 'E'.\n"
                    f"In row {e_row}, the line is: '{e_line}'\n"
                    f"'E' is at row={e_row}, col={e_col} (the {e_col}th character in the row).\n"
                )
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
    # Scale n_points based on maze size - more points for larger mazes
    n_points = max(5, maze_size * 2)  # Minimum 5 points, scales up with maze size
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

def dict_to_prompt_completion(ex):
    """Convert a single maze example into multiple training instances for different tasks."""
    maze = ex["maze"]
    maze_id = ex.get("id", "unknown")
    chain_of_thought = ex.get("chain_of_thought", {})
    training_examples = []

    # 1. Start/End Detection Task
    start = ex["answer"]["start"]
    end = ex["answer"]["end"]
    start_end_prompt = (
        "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
        "Return only these keys: `start` (0-based [row,col]), `end` (0-based [row,col]), and `think` (string)\n\n"
        f"<maze>\n{maze}\n</maze>\n\n"
        "Identify the coordinates of start 'S' and end 'E' in the maze."
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

    # 2. Available Directions Tasks - for multiple points
    direction_samples = ex.get("direction_samples", [])
    for idx, sample in enumerate(direction_samples):
        position = sample["position"]
        directions = sample["walkable_directions"]
        dir_prompt = (
            "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
            "Return only these keys: `available_directions` (array using 'up','down','left','right') and `think` (string)\n\n"
            f"<maze>\n{maze}\n</maze>\n\n"
            f"From position {position}, list all available directions where movement is possible."
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

    # 3. Valid Move Tasks
    move_samples = ex.get("move_samples", [])
    for idx, sample in enumerate(move_samples):
        move_prompt = (
            "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
            "Return only these keys: `is_valid` (boolean) and `think` (string)\n\n"
            f"<maze>\n{maze}\n</maze>\n\n"
            f"From position {sample['position']}, is moving {sample['move']} valid?"
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

    # 4. Optimal Next Step Tasks
    solution_stages = ex.get("stages", [])
    for idx, stage in enumerate(solution_stages):
        step_prompt = (
            "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
            "Return only these keys: `optimal_step` (one of: 'up','down','left','right') and `think` (string)\n\n"
            f"<maze>\n{maze}\n</maze>\n\n"
            f"From position {stage['position']}, what is the optimal next step to reach the goal?"
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
    # 5. Full Solution Task
    if solution_stages:
        # Build thinking block
        full_solution_thinking = chain_of_thought.get('start_end', '') + "\n\nPath finding steps:\n"
        path_sequence = []
        
        for stage in solution_stages:
            full_solution_thinking += stage.get('reasoning', '') + "\n"
            path_sequence.append(stage["optimal_step"])
            
        solution_prompt = (
            "You are a maze assistant. Read the ASCII maze and solve it step by step. Answer in STRICT JSON.\n"
            "Return only these keys: `path` (array of directions), `think` (detailed reasoning)\n\n"
            "Walk through the maze from start 'S' to end 'E', listing each step of the solution.\n"
            "Consider walls, optimal path, and explain your thinking process.\n\n"
            f"<maze>\n{maze}\n</maze>\n\n"
            "Provide the complete solution path with detailed reasoning."
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
    messages = [
        {"role": "system", "content": "You are a maze-solving assistant. Analyze ASCII mazes to find start/end points, determine valid moves, list available directions, and provide optimal next steps. Always respond in strict JSON format with only the requested fields. Include step-by-step reasoning in the 'think' field when provided."},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_target(answer_start: List[int], answer_dirs: List[str]) -> str:
    # The trainer will learn only from these target tokens (prompt is masked out).
    return json.dumps({
        "start": [int(answer_start[0]), int(answer_start[1])],
        "available_directions": list(answer_dirs)
    }, ensure_ascii=False)

def clamp_and_pad(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) > max_len:
        # For chat SFT, truncating the left (prompt side) is usually safer than chopping off the label.
        # But since we create the full sequence ourselves, keep it simple: right-truncate.
        ids = ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))

def filter_jsonl_by_task_ratio(input_data, task_ratios: Dict[str, float]) -> str:
    """
    Filter JSONL file to keep specified ratios of each task.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        task_ratios: Dictionary mapping task names to desired ratios (0-1)
        
    Returns:
        String containing filtered JSONL content
    """
    # Validate ratios sum to 1
    if abs(sum(task_ratios.values()) - 1.0) > 0.001:
        raise ValueError("Task ratios must sum to 1.0")

    # Read all examples
                
    # Group by task
    task_groups = {}
    for ex in input_data:
        task = ex["task"]
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(ex)
        
    # Calculate counts to keep for each task
    total_examples = len(input_data)
    keep_counts = {
        task: int(ratio * total_examples) 
        for task, ratio in task_ratios.items()
    }
    
    # Sample examples to keep
    filtered_examples = []
    for task, count in keep_counts.items():
        if task in task_groups:
            task_examples = task_groups[task]
            samples = random.sample(task_examples, min(count, len(task_examples)))
            filtered_examples.extend(samples)
            
    # Convert back to JSONL
    return "\n".join(json.dumps(ex) for ex in filtered_examples) + "\n"


# Update dict_to_prompt_completion to include full solution
def dict_to_prompt_completion(ex):
    # Keep existing code...
    result = dict_to_prompt_completion_original(ex)  # Store original function output
    
    # Add full solution task
    full_solution = add_full_solution_to_dict_prompt(ex)
    if full_solution:
        result += full_solution
        
    return result