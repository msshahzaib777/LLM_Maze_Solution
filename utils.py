def clean_maze_ascii(maze_ascii: str) -> str:
    return maze_ascii.replace("X", " ")

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
    maze_ascii = clean_maze_ascii(m.as_ascii())
    start = find_start(maze_ascii)
    surroundings = get_surroundings(maze_ascii, start)

    # "Chain of thought"-style explanation
    reasoning = (
        f"I scan the maze for the symbol 'S'.\n"
        f"I found 'S' at row={start[0]}, col={start[1]}.\n"
        f"Now I check each neighbor:\n"
        + "\n".join([f"- {d}: {'open' if v else 'wall'}" for d,v in surroundings.items()])
    )

    return {
        "maze": maze_ascii,
        "prompt": "Identify the start location and its available directions in this maze.",
        "chain_of_thought": reasoning,
        "answer": {
            "start": start,
            "available_directions": [d for d,v in surroundings.items() if v]
        }
    }
