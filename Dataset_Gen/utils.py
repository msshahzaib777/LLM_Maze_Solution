import json
from mlx_lm import load, generate
from typing import List, Dict, Any, Optional, Tuple

_, tokenizer = load("nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx")

def clean_maze_ascii(maze_ascii: str) -> str:
        return maze_ascii.replace("+", " ")

def find_start(maze_ascii: str):
    lines = maze_ascii.splitlines()
    for i, row in enumerate(lines):
        for j, c in enumerate(row):
            if c == "S":
                return (i, j)
    return None

def get_surroundings(maze_ascii: str, start):
    lines = maze_ascii.splitlines()
    i, j = start
    directions = {}
    moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    for d, (di, dj) in moves.items():
        ni, nj = i + di, j + dj
        if 0 <= ni < len(lines) and 0 <= nj < len(lines[0]):
            directions[d] = (lines[ni][nj] != "#")  # True if open, False if wall
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

    return {
        "maze": maze_ascii,
        "prompt": "Identify the start location which is labelled with S in this maze.",
        "chain_of_thought": reasoning,
        "answer": {
            "start": start,
            "available_directions": [d for d,v in surroundings.items() if v]
        },
        "solved_maze": solved_maze,
        "maze_size": (m.grid.shape[0] - 1)/2
    }

def dict_to_prompt_completion(dict, start=True, available_directions=True, thinking=False):
    maze = dict["maze"]
    prompt = dict["prompt"]
    cot = dict["chain_of_thought"]
    answer = ""
    if start:
        if available_directions:
            answer = json.dumps(dict["answer"])
            cot = dict["chain_of_thought"]["start"] + dict["chain_of_thought"]["available_directions"]
        else:
            answer = dict["answer"]["start"]
            cot = dict["chain_of_thought"]["start"]
    system = '''You are an expert Maze Solver. 
                Your task is to analyze the given maze (ASCII grid) and answer questions about it. 
                The maze always contains:
                - a start position, marked with 'S'
                - an exit position, marked with 'E'
                - walls '#'
                - open paths ' '
                
                When answering:
                - Always return the result in the requested format (e.g., coordinates as [row, col]).
                - Use 0-based indexing for rows and columns unless instructed otherwise.
                - Do not include extra text beyond the answer unless explicitly asked.
                - If something is missing or invalid in the maze, state it clearly.'''
    input_text = (f"Maze:\n{maze}\n\nTask: {prompt}\n")
    completion_text = ""
    if thinking:
        completion_text = f"<think>\n{cot}\n"
    completion_text = completion_text +  f"</think>{answer}\n<|im_end|>"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": input_text}]
    # --- Load fused model + tokenizer ---
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return json.dumps({
                "prompt": text,
                "completion": completion_text
            }) + "\n"


# --------------------------
# Dataset: JSONL → supervised pairs
# --------------------------
def build_prompt(maze_ascii: str, user_prompt: str) -> str:
    # Keep formatting consistent so tokenization is stable.
    user = (
        "You are a maze assistant. Read the ASCII maze and answer in STRICT JSON.\n"
        "Return only these keys: `start` (0-based [row,col]) and `available_directions` "
        "(array using 'up','down','left','right').\n\n"
        "<maze>\n" + maze_ascii + "\n</maze>\n\n"
        + user_prompt.strip()
    )
    messages = [
        {"role": "system", "content": "Follow the schema exactly. No extra text."},
        {"role": "user", "content": user},
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
