#!/usr/bin/env python3
"""Convert raw maze records into the unified JSONL format used for training."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from Dataset_Gen.utils import (
    available_direction_enums,
    apply_direction,
    find_end,
    find_start,
    is_walkable as maze_is_walkable,
    map_text_dirs_to_enum,
)

TASKS = ("DETECT_START_END", "AVAILABLE_DIRECTIONS", "VALID_MOVE")
NEG_MOVES_PER_RECORD = 4
DIRS = ("U", "D", "L", "R")
WALKABLE_CHARS = (" ", "S", "E")


def read_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    

def to_text_completion(sample: Dict[str, Any]) -> Dict[str, str]:
    messages = sample.get("messages", [])
    system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
    parts = []
    if system_msg:
        parts.append(f"[SYSTEM]\n{system_msg}")
    if user_msg:
        parts.append(f"[USER]\n{user_msg}")
    text = "\n\n".join(parts)
    targets = sample.get("supervision", {}).get("targets", {})
    completion = json.dumps(targets, ensure_ascii=False)
    return {"text": text, "completion": completion}


def maze_size(maze: str) -> Tuple[int, int]:
    lines = maze.splitlines()
    h = len(lines)
    w = max((len(line) for line in lines), default=0)
    return h, w


def base_sample(rec_id: str, maze: str, split: str, difficulty: str) -> Dict[str, Any]:
    h, w = maze_size(maze)
    return {
        "id": rec_id,
        "version": "v1.0",
        "difficulty": difficulty,
        "meta": {"split": split},
        "maze": {"repr": maze, "height": h, "width": w, "walkable_chars": list(WALKABLE_CHARS)},
        "io_format": "chat",
    }


def sample_detect_start_end(base: Dict[str, Any], start: Tuple[int, int], end: Tuple[int, int]) -> Dict[str, Any]:
    sample = dict(base)
    sample["task_type"] = "DETECT_START_END"
    sample["messages"] = [
        {"role": "system", "content": "Use 0-indexed rows/cols with U,D,L,R directions."},
        {
            "role": "user",
            "content": (
                "Return JSON for start and end coordinates of 'S' and 'E' respectively.\n\n" + sample["maze"]["repr"]
            ),
        },
    ]
    sample["supervision"] = {
        "targets": {
            "start": {"row": start[0], "col": start[1]},
            "end": {"row": end[0], "col": end[1]},
        }
    }
    return sample


def sample_available_dirs(base: Dict[str, Any], at: Tuple[int, int], dirs: List[str]) -> Dict[str, Any]:
    sample = dict(base)
    sample["task_type"] = "AVAILABLE_DIRECTIONS"
    sample["messages"] = [
        {"role": "system", "content": "Use 0-indexed rows/cols with U,D,L,R directions."},
        {
            "role": "user",
            "content": (
                f"At coordinate {{'row':{at[0]},'col':{at[1]}}}, list walkable directions (U,D,L,R).\n\n"
                + sample["maze"]["repr"]
            ),
        },
    ]
    sample["supervision"] = {
        "targets": {
            "at": {"row": at[0], "col": at[1]},
            "available_dirs": dirs,
        }
    }
    return sample


def sample_valid_move(base: Dict[str, Any], origin: Tuple[int, int], direction: str, valid: bool, dest=None) -> Dict[str, Any]:
    sample = dict(base)
    sample["task_type"] = "VALID_MOVE"
    sample["messages"] = [
        {"role": "system", "content": "Use 0-indexed rows/cols with U,D,L,R directions."},
        {
            "role": "user",
            "content": (
                f"From {{'row':{origin[0]},'col':{origin[1]}}}, is moving '{direction}' valid?"
                " Answer YES or NO and share the new coordinate if valid."
            ),
        },
    ]
    targets = {
        "from": {"row": origin[0], "col": origin[1]},
        "dir": direction,
        "valid": bool(valid),
    }
    if valid and dest is not None:
        targets["to"] = {"row": dest[0], "col": dest[1]}
    sample["supervision"] = {"targets": targets}
    return sample


def infer_difficulty(height: int, width: int, path_len: int) -> str:
    area = height * width
    if area <= 100 or path_len <= 12:
        return "easy"
    if area <= 225 or path_len <= 30:
        return "med"
    return "hard"


def record_samples(rec: Dict[str, Any], idx: int, split: str) -> List[Dict[str, Any]]:
    maze = rec.get("maze", "")
    if not maze.strip():
        return []

    start = tuple(rec.get("answer", {}).get("start", ())) or find_start(maze)
    end = tuple(rec.get("answer", {}).get("end", ())) or find_end(maze)
    if not start or not end:
        return []

    height, width = maze_size(maze)
    dirs_enum = map_text_dirs_to_enum(rec.get("directions", []))
    runs = len(dirs_enum)
    difficulty = infer_difficulty(height, width, runs)

    base = base_sample(f"rec{idx}", maze, split, difficulty)

    samples: List[Dict[str, Any]] = []
    if "DETECT_START_END" in TASKS:
        samples.append(sample_detect_start_end(base, start, end))

    if "AVAILABLE_DIRECTIONS" in TASKS:
        provided = map_text_dirs_to_enum(rec.get("answer", {}).get("available_directions", []))
        dirs = available_direction_enums(maze, start) or provided
        samples.append(sample_available_dirs(base, start, dirs))

    if "VALID_MOVE" in TASKS and dirs_enum:
        cursor = start
        for d in dirs_enum:
            nxt = apply_direction(cursor, d)
            is_valid = maze_is_walkable(maze, nxt)
            samples.append(sample_valid_move(base, cursor, d, is_valid, nxt if is_valid else None))
            cursor = nxt if is_valid else cursor

        visited = [start]
        cursor = start
        for d in dirs_enum[: max(1, len(dirs_enum) // 3)]:
            cursor = apply_direction(cursor, d)
            visited.append(cursor)
        random.shuffle(visited)
        added = 0
        for pos in visited:
            illegal = [d for d in DIRS if d not in available_direction_enums(maze, pos)]
            if not illegal:
                continue
            bad_dir = random.choice(illegal)
            samples.append(sample_valid_move(base, pos, bad_dir, False))
            added += 1
            if added >= NEG_MOVES_PER_RECORD:
                break

    return samples


def convert(src: Path, split: str) -> List[Dict[str, Any]]:
    rows = read_records(src)
    samples: List[Dict[str, Any]] = []
    for idx, rec in enumerate(rows):
        samples.extend(record_samples(rec, idx, split))
    samples.sort(key=lambda x: (x["id"], x["task_type"]))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify maze datasets into the training schema.")
    parser.add_argument("--in", dest="inp", required=True, help="Input JSON/JSONL")
    parser.add_argument("--out", dest="outp", required=True, help="Output JSONL path for structured samples")
    parser.add_argument(
        "--out-text",
        dest="out_text",
        help="Optional JSONL path for text/completion pairs (defaults to <out>_pairs.jsonl)",
    )
    parser.add_argument("--split", default="train", help="Dataset split label")
    args = parser.parse_args()

    structured_path = Path(args.outp)
    text_pairs_path = (
        Path(args.out_text)
        if args.out_text
        else structured_path.with_name(structured_path.stem + "_pairs.jsonl")
    )

    samples = convert(Path(args.inp), args.split)
    write_jsonl(structured_path, samples)
    write_jsonl(text_pairs_path, [to_text_completion(s) for s in samples])
    print(
        f"Wrote {len(samples)} structured samples to {structured_path} and text/completion pairs to {text_pairs_path}"
    )


if __name__ == "__main__":
    main()
