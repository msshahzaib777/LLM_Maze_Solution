import json
from mlx_lm import load, generate
from typing import List, Dict, Any, Optional, Tuple
import random

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
    maze = maze_ascii
    start = find_start(maze)
    current = start
    stages = []
    while True:
        # Exit if we reached the end
        if maze_value_at(maze, current) == "E":
            break
            
        # Get next optimal direction
        direction = get_correct_direction_at(maze, current, maze_size)
        if not direction:
            break
            
        # Calculate next position
        next_pos = apply_direction(current, direction[0])
        
        # Build reasoning for this stage
        reasoning = (
            f"At position (row={current[0]}, col={current[1]}).\n"
            f"Checking possible directions...\n"
            f"Optimal direction is '{direction[0]}' to move towards the goal.\n"
            f"Moving {direction[0]} by vector {direction[1]} to position (row={next_pos[0]}, col={next_pos[1]}).\n"
            f"Marking current position with '-' to track the path."
        )
            
        # Store current stage info with path and reasoning
        stage = {
            "position": current,
            "optimal_step": direction[0],
            "maze_before": maze,
            "maze_after": update_maze(maze, current, "-"),
            "path": direction[0],
            "reasoning": reasoning
        }
        stages.append(stage)
        
        # Move to next position
        maze = stage["maze_after"]
        current = next_pos
        
        return stages

def generate_chain_of_thought(tasks: List[str], maze_ascii: str, start: Optional[Tuple[int, int]] = None,
                            end: Optional[Tuple[int, int]] = None,
                            surroundings: Optional[Dict[str, bool]] = None,
                            move: Optional[str] = None,
                            optimal_step: Optional[str] = None) -> Dict[str, str]:
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
            reasoning["start"] = reasoning_text
            reasoning["end"] = reasoning_text

        elif task == "AVAILABLE_DIRECTIONS":
            if start is not None and surroundings is not None:
                s_row, s_col = start
                reasoning_text = (
                    f"I am at position (row={s_row}, col={s_col}).\n"
                    f"I check each neighbor cell to see if it's open or a wall:\n"
                )
                for d, v in surroundings.items():
                    reasoning_text += f"- {d}: {'open' if v else 'wall'}\n"
                reasoning["available_directions"] = reasoning_text
    return reasoning

def make_training_example(m, tasks: List[str] = ["DETECT_START_END"]):
    solved_maze = str(m)
    maze_ascii = clean_maze_ascii(m.tostring(True, True))
    start = m.start
    end = getattr(m, "end", None)
    surroundings = get_surroundings(maze_ascii, start)
    maze_size = int((m.grid.shape[0] - 1)/2)
    solution = get_solution(maze_ascii, maze_size)
    
    return {
        "maze": maze_ascii,
        "prompt": "Identify the start location which is labelled with S in this maze.",
        "chain_of_thought": generate_chain_of_thought(
            tasks=tasks,
            maze_ascii=maze_ascii,
            start=start,
            end=end,
            surroundings=surroundings,
            optimal_step=solution[0]["path"] if solution else None
        ),
        "answer": {
            "start": start,
            "available_directions": [d for d, v in surroundings.items() if v],
            "end": end,
        },
        "stages": solution,  # Includes stages with reasoning
        "solved_maze": solved_maze,
        "maze_size": maze_size
    }

def dict_to_prompt_completion(ex, target_mode):
    maze = ex["maze"]
    user_prompt = ex.get("prompt", "Identify the co ordinated of start 'S' and end 'E' location and available directions in this maze.")
    # Ground truth:
    ans = ex.get("answer", {})
    start = ans.get("start", None)
    dirs = ans.get("available_directions", None)

    
    target_text = ""
    if target_mode == "start_only":
        user_prompt = (
        "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
        "Return only these keys: `start` (0-based [row,col]).\n\n"
        "<maze>\n" + maze + "\n</maze>\n\n"
        + "Identify the co ordinated of start 'S' in the maze."
    )
        target_obj = {"start": start}
        target_text = json.dumps(target_obj, ensure_ascii=False)
    elif target_mode == "start_available_direction":
        user_prompt = (
        "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
        "Return only these keys: `start` (0-based [row,col]) and `available_directions` "
        "(array using 'up','down','left','right').\n\n"
        "<maze>\n" + maze + "\n</maze>\n\n"
        + "Identify the co ordinated of start 'S' in the maze."
    )
        target_text = build_target(start, dirs)
    elif target_mode == "start_and_end":
        user_prompt = (
        "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
        "Return only these keys: `start` (0-based [row,col]) and `end` (0-based [row,col])\n\n"
        "<maze>\n" + maze + "\n</maze>\n\n"
        + "Identify the co ordinated of start 'S' in the maze."
    )
        end = ans.get("end", None)
        target_obj = {"start": start, "end": end}
        target_text = json.dumps(target_obj, ensure_ascii=False)
    elif target_mode == "optimal_next_step":
        pass
    # Build strings
    prompt_text = build_prompt(maze, user_prompt)

    return(json.dumps({
        "prompt": prompt_text,
        "completion": target_text
    }) + "\n")


# --------------------------
# Dataset: JSONL → supervised pairs
# --------------------------
def build_prompt(maze_ascii: str, user_prompt: str) -> str:
    # Keep formatting consistent so tokenization is stable.
    messages = [
        {"role": "system", "content": "Follow the schema exactly. No extra text."},
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