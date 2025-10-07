# import json

# from Dataset_Gen.utils import find_start, get_surroundings

# def get_correct_direction_at(maze: str, position, maze_size):
#     i, j = position
#     moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#     for d, (di, dj) in moves.items():
#         ni, nj = i + di, j + dj
#         if 0 <= ni <= maze_size and 0 <= nj <= maze_size:
#             if (maze_value_at(maze, (ni, nj)) in ["+", "E"]):  # True if +, else False
#                 return d, (di, dj)

# def update_maze(maze, position, new_value):
#     lines = maze.splitlines()
#     new_line = list(lines[position[0]])
#     new_line[position[1]] = new_value
#     lines[position[0]] = "".join(new_line)
#     return "\n".join(lines)

# def maze_value_at(maze, position):
#     lines = maze.splitlines()
#     new_line = list(lines[position[0]])
#     return new_line[position[1]]

# def get_solution(sample):
#     maze = sample["solved_maze"]
#     start = find_start(maze)
#     not_end = True
#     current = start
#     directions = []
#     stages = [maze]
#     while not_end:
#         if maze_value_at(maze, current) == "E":
#             maze = update_maze(maze, current, "-")
#             break
#         surroundings = get_correct_direction_at(maze, current, int(ds_dict[0]["maze_size"] * 2))
#         maze = update_maze(maze, current, "-")
#         stages.append(maze)
#         directions.append(surroundings[0])
#         current = current[0] + surroundings[1][0], current[1] + surroundings[1][1]
#     return directions, stages
# dataset_json = "data/maze_training_4.json"
# ds_dict = json.load(open(dataset_json))

# sample = ds_dict[0]
# print(get_solution(sample))


from mazelib import Maze
from mazelib.generate.Prims import Prims

m = Maze()
m.generator = Prims(7, 7)
m.generate()
print(m.generator.H)
print(m.generator.W)
print(m.generator.h)
print(m.generator.w)    