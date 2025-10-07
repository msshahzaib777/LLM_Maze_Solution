# Repository Guidelines
LLM Maze Solution fine-tunes Qwen-family models with MLX to decode maze layouts. Follow these guidelines to keep contributions aligned.

## Project Structure & Module Organization
- `main.py` drives distributed LoRA training and gradient synchronization.
- `Dataset_Gen/` hosts dataset builders (`datasetGeneration.py`, `testSetGen.py`) and schema helpers.
- `classes/` defines reusable dataset wrappers, loaders, and metrics consumed by training and evaluation.
- `finetuned_model/` stores adapters and evaluation outputs; keep large checkpoints out of Git.
- `data/` holds JSONL splits such as `custom__start_and_end_1`; `logs/` and `plots/` track experiment artifacts.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` prepares a local env.
- `pip install -U mlx mlx-vlm datasets pillow opencv-python soundfile` installs required packages; add `ffmpeg` on PATH for video/audio work.
- `python main.py --model <id> --data data/custom__start_and_end_1 --adapter-path finetuned_model/adapters_dir_start_end` launches fine-tuning.
- `python finetunedInference.py --adapter finetuned_model/...` runs inference on existing outputs.
- `python finetunedTest.py` batches evaluation and writes metrics under `finetuned_model/adapters_dir_start_end/eval_*`.

## Coding Style & Naming Conventions
Use PEP 8 Python style with 4-space indentation and descriptive snake_case for functions, PascalCase for classes (see `MazeJSONLDataset`), and uppercase constants. Keep functions pure where possible and add type hints for new APIs. Reuse helper utilities in `Dataset_Gen/utils.py` instead of duplicating logic.

## Testing Guidelines
Prefer dataset-backed checks: run `python finetunedTest.py` after changes to confirm start/end predictions and directional metrics. For dataset tweaks, regenerate fixtures via `python Dataset_Gen/datasetGeneration.py` and spot-check the JSONL output. Document edge cases in `logs/` or brief Markdown notes when manual verification is required.

## Commit & Pull Request Guidelines
Write concise, present-tense commit subjects (~50 chars) mirroring existing history (e.g., `cache_inference`, `testing adding end co-ordinates`). Include context in the body when touching training configs or data formats. PRs should describe motivation, outline command sequences used for validation, and link datasets or experiment IDs. Attach before/after metrics or sample predictions when behavior changes.
