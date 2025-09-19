from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.tokenization.modular.maze_tokenizer_modular import MazeTokenizerModular

cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test", # name is only for you to keep track of things
    grid_n=5, # number of rows/columns in the lattice
    n_mazes=1, # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

mazeTokenizer = MazeTokenizerModular()
list_of_list = dataset[0].as_tokens(mazeTokenizer)

#
# # --- Build dataset ---
# examples = [make_training_example(m) for m in dataset]
#
# # --- Save to JSON ---
# with open("maze_training.json", "w") as f:
#     json.dump(examples, f, indent=2)
#
# print("✅ Saved", len(examples), "examples to maze_training.json")
#


