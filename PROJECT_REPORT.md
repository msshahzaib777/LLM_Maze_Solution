# LLM_Maze_Solution — Project Report

Date: 2025-12-04
Branch: gpu_pytroch

## 1) Overview
This repository explores teaching and evaluating LLMs to navigate ASCII mazes. The work covers dataset generation, training prompts/targets, and multiple inference strategies for stepwise maze solving with safety constraints and logging.

Core ideas:
- Synthetic maze generation with ground-truth solutions.
- Supervised datasets for several tasks (detect start/end, available directions, valid move, optimal next step, full solution).
- An environment→agent loop for inference (and an alternative dataset in Dataset_Gen2) that feeds current state and short history to the LLM, validates the returned move, and repeats until success or bounded failure.

## 2) Code Structure (high-level)
Top-level files of interest:
- `mazeSolution.py` — Step-by-step maze solver using a fine-tuned/base model. Implements local view, constraints, metrics, and streaming logs.
- `Dataset_Gen/` — Original dataset pipeline producing multi-task JSONL (DETECT_START_END, AVAILABLE_DIRECTIONS, VALID_MOVE, OPTIMAL_NEXT_STEP, MAZE_SOLUTION).
  - `datasetGeneration.py` — Generates mazes, splits into train/valid/test, materializes JSONL.
  - `utils.py` — Prompt building, maze utilities, and task-specific example formatting.
- `Dataset_Gen2/` — New environment–agent dataset pipeline focused on OPTIMAL_NEXT_STEP with state+history per step.
  - `datasetGeneration.py` — Mirrors original structure but emits env–agent samples.
  - `utils.py` — Env runner that enforces visit and global budget constraints; formats prompts accordingly.
  - `README.md` — How it differs and how to run.
- `constant.py` — Paths and training/runtime configuration (models, adapters, tokens, dataset name, etc.).
- Training/analysis helpers — `ddpTraining.py`, `mlxTrainingScript.py`, `finetunedInference.py`, `finetunedTest.py`, `plot_learning_curves.py`, `calculate_accuracy.py`, etc.
- `logs/` — Run logs for solver and experiments.

## 3) Datasets

### 3.1 Original (Dataset_Gen)
- Generator: `mazelib` with Prim’s algorithm and `BacktrackingSolver` for ground truth.
- Exports multiple tasks for each maze via `utils.dict_to_prompt_completion()`:
  - DETECT_START_END — locate S and E.
  - AVAILABLE_DIRECTIONS — legal moves from a position.
  - VALID_MOVE — boolean validity per candidate move.
  - OPTIMAL_NEXT_STEP — the next step along the ground-truth path.
  - MAZE_SOLUTION — full path as a sequence.
- Splitting: stratified by `maze_size` using `train_test_split` with configurable ratios.
- Length estimation: computes recommended max sequence length per split.

Key config example (in `datasetGeneration.py`):
- Monte-carlo parameters for diversity.
- Maze sizes and counts.
- Task ratios for down-selection into JSONL.

### 3.2 Env–Agent Dataset (Dataset_Gen2)
Motivation: align data with runtime behavior (state→action loop).
- Single task: OPTIMAL_NEXT_STEP.
- Each sample encodes: current position, goal (E), short move history, current maze state (visited cells marked), and ground-truth next step.
- Constraints during sample creation:
  - Per-cell visit cap: max 10 visits.
  - Global move budget: `maze_size^2 * 2` (models typically fail via loops before hitting this).
- Structure mirrors `Dataset_Gen` for low friction: same generation entrypoint, splitting, and a summarizer for sequence lengths.

## 4) Prompts and Targets
- Chat template assembled via `Dataset_Gen/utils.build_prompt()` to ensure training and inference use identical system+user formatting.
- OPTIMAL_NEXT_STEP prompt (training and inference):
  "From {position}, what's the optimal next step to reach goal? Return JSON with keys: optimal_step (direction). Minimize reasoning and thinking. <maze>…</maze>"
- Env–Agent (Dataset_Gen2) adds: short history + local/visited view while preserving strict-JSON response.

## 5) Inference: `mazeSolution.py`
Solver loop (model-agnostic):
1. Generate a maze using the same pipeline as the dataset.
2. Build the OPTIMAL_NEXT_STEP prompt (plus local context) with the exact same `build_prompt()` used in training.
3. Generate model output; robustly extract `optimal_step` from JSON.
4. Validate move against immediate surroundings; update the maze visualization and internal state; repeat.

Implemented constraints & metrics:
- Illegal move handling: count and stop after 5 invalid/no-direction events.
- Visit limit per cell: warn and, if exceeded, stop with `visit_limit_exceeded`.
- Global move budget: defensive bound based on maze size.
- Success metrics across N mazes: success rate, average steps, average max visit count, and per-failure reason percentages.

Streaming logs:
- Minimal `Tee` redirection writes to both console and timestamped file under `logs/` in real time.

## 6) Configuration (`constant.py`)
Paths:
- `ROOT_DIR`, `MODEL_DIR`, `ADAPTER_DIR`, `MERGED_MODEL_DIR`, `DATA_DIR`.
Model/training runtime:
- `base_id` (e.g., Qwen/Qwen3-14B).
- `ADAPTER_NAME`, `step_num`, quantization toggle, batch/accum, LR, tokens.
- `INF_MAX_NEW_TOKENS` controls inference generation length; solver uses this via `mazeSolution.py`.
- `Dataset_NAME` defaults to `env_agent_v1` to align with Dataset_Gen2.

## 7) Experiments & Outcomes (what changed)
- Reused `build_prompt()` at inference to guarantee template parity with training.
- Added a compact `solve_maze()` with:
  - Local view around the agent and distances to walls to provide helpful context.
  - Validation of suggested moves, with early-exit policies and categorized failure reasons:
    - `no_direction_extracted`, `invalid_moves`, `visit_limit_exceeded`, `global_budget_exceeded`, `max_steps_reached`.
  - Summary over multiple mazes including success rate and failure percentages.
- Introduced `Dataset_Gen2` for env–agent step supervision with minimal structural divergence from the original pipeline.
- Fixed cross-folder imports so `Dataset_Gen/datasetGeneration.py` works both from project root and when executed inside `Dataset_Gen`.
- Added streaming log capture to `mazeSolution.py` using a minimal `Tee` construct.

## 8) How to Run
Environment (example):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generate datasets:
```bash
# Original multi-task
python -m Dataset_Gen.datasetGeneration

# Env–Agent dataset
python -m Dataset_Gen2.datasetGeneration
```

Solve mazes with a model:
```bash
python mazeSolution.py --maze_size 5 --num_mazes 10
# Logs stream to console and to logs/maze_solution_YYYYMMDD_HHMMSS.log
```

Notes:
- The solver loads base + (optional) adapter per `constant.py`.
- MAX/INF token limits are taken from constants.

## 9) Next Steps (suggested)
- Add curriculum scheduling over maze sizes during training to improve generalization.
- Integrate beam search or value-guided decoding for inference stability.
- Enrich state summary with shortest-path distance estimates (learned heuristic) to reduce loops.
- Add tests for JSON extraction and move validation edge cases.
- Log compact JSON traces per episode for downstream analysis.
