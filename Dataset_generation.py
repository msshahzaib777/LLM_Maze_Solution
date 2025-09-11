import json

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

from utils import make_training_example

cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test", # name is only for you to keep track of things
    grid_n=5, # number of rows/columns in the lattice
    n_mazes=100000, # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)


# --- Build dataset ---
examples = [make_training_example(m) for m in dataset]

# --- Save to JSON ---
with open("maze_training.json", "w") as f:
    json.dump(examples, f, indent=2)

print("✅ Saved", len(examples), "examples to maze_training.json")

