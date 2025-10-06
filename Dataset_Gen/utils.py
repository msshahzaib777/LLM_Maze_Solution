import json
from mlx_lm import load, generate
from typing import List, Dict, Any, Optional, Tuple

_, tokenizer = load("nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx")

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


def make_training_example(m):
    solved_maze = str(m)
    maze_ascii = clean_maze_ascii(m.tostring(True, True))
    start = m.start
    surroundings = get_surroundings(maze_ascii, start)
    reasoning = {
        "start": (f"I scan the maze for the symbol 'S'.\n"
            f"I found 'S' at row={start[0]}, col={start[1]}.\n"),
        "available_directions": ( f"Now I check each neighbor:\n"
        + "\n".join([f"- {d}: {'open' if v else 'wall'}" for d,v in surroundings.items()]))
    }
    direction, stages = get_solution({"solved_maze": solved_maze, "maze_size": (m.grid.shape[0] - 1)/2 })
    return {
        "maze": maze_ascii,
        "prompt": "Identify the start location which is labelled with S in this maze.",
        "chain_of_thought": reasoning,
        "answer": {
            "start": start,
            "available_directions": [d for d, v in surroundings.items() if v],
            "end": m.end
        },
        "solved_maze": solved_maze,
        "directions": direction,
         "stages": stages,
        "maze_size": (m.grid.shape[0] - 1)/2
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

def get_solution(sample):
    maze = sample["solved_maze"]
    start = find_start(maze)
    not_end = True
    current = start
    directions = []
    stages = [maze]
    while not_end:
        if maze_value_at(maze, current) == "E":
            maze = update_maze(maze, current, "-")
            break
        surroundings = get_correct_direction_at(maze, current, int(sample["maze_size"] * 2))
        maze = update_maze(maze, current, "-")
        stages.append(maze)
        directions.append(surroundings[0])
        current = apply_direction(current, surroundings[0])
    return directions, stages
