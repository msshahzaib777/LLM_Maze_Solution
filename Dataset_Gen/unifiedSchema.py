#!/usr/bin/env python3
"""
maze_dataset_unifier.py

Transform your current maze dataset format into the unified JSONL schema:
- DETECT_START_END
- AVAILABLE_DIRECTIONS
- VALID_MOVE (positives from your path + a few negatives)
- ALL_IN_ONE (with compressed runs)

Supports input as:
- a JSON array file (one big list), or
- a JSONL file (one object per line)

Usage:
    python maze_dataset_unifier.py --in data_raw.jsonl --out maze_unified.jsonl --split train

Notes:
- Coordinates are assumed 0-indexed, origin at top-left (row down, col right).
- Walls: '#'
- Walkable in raw mazes: space ' ', 'S', 'E' (you can add '.')
- We keep your ASCII maze exactly as given; we just declare walkable chars.
"""

import argparse, json, os, sys, random
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ------------- CONFIG YOU MAY TWEAK -------------
EMIT_TASKS = {
    "DETECT_START_END": True,
    "AVAILABLE_DIRECTIONS": True,
    "VALID_MOVE": True,
    "ALL_IN_ONE": False,
}

# how many negative VALID_MOVE examples to try per record
NEG_MOVES_PER_RECORD = 4

# scratchpad policy
SCRATCHPAD_POLICY = "include_for_training_but_mask_at_inference"  # or "omit_at_inference"

# direction enum + deltas
DIRS = ["U", "D", "L", "R"]
DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

# treat these as walkable in the *raw* maze representation
WALKABLE_CHARS = [" ", "S", "E"]

# curriculum / difficulty heuristics
def infer_difficulty(h: int, w: int, path_len: int) -> str:
    area = h * w
    if area <= 100 or path_len <= 12:
        return "easy"
    if area <= 225 or path_len <= 30:
        return "med"
    return "hard"

# ------------- IO HELPERS -------------

def read_any_jsonlike(path: Path) -> List[Dict[str, Any]]:
    """Read either a JSON array file or JSONL file into a list of dicts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # try JSON array first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        # if it's a single object, wrap it
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    # fallback: JSONL
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------- MAZE GEOMETRY -------------

def parse_maze(maze_str: str) -> Tuple[List[str], int, int]:
    lines = maze_str.splitlines()
    h = len(lines)
    w = max(len(row) for row in lines) if lines else 0
    return lines, h, w

def in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w

def char_at(grid: List[str], r: int, c: int) -> str:
    row = grid[r]
    return row[c] if 0 <= c < len(row) else "#"  # treat missing as wall

def is_walkable(ch: str) -> bool:
    return ch in WALKABLE_CHARS

def find_S_E(grid: List[str]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    s = e = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                s = (r, c)
            elif ch == "E":
                e = (r, c)
    return s, e

def available_dirs_at(grid: List[str], r: int, c: int) -> List[str]:
    out = []
    h, w = len(grid), max(len(row) for row in grid) if grid else 0
    for d in DIRS:
        dr, dc = DELTA[d]
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, h, w) and is_walkable(char_at(grid, nr, nc)):
            out.append(d)
    return out

def apply_dir(pos: Tuple[int,int], d: str) -> Tuple[int,int]:
    dr, dc = DELTA[d]
    return pos[0] + dr, pos[1] + dc

def compress_runs(dirs: List[str]) -> List[Dict[str, Any]]:
    """Convert ['L','L','U','U','R'] -> [{'dir':'L','steps':2}, ...]"""
    if not dirs:
        return []
    runs = []
    cur = dirs[0]
    count = 1
    for d in dirs[1:]:
        if d == cur:
            count += 1
        else:
            runs.append({"dir": cur, "steps": count})
            cur, count = d, 1
    runs.append({"dir": cur, "steps": count})
    return runs

def map_text_dirs_to_enum(text_dirs: List[str]) -> List[str]:
    """Your dataset uses 'left','right','up','down'. Map to 'L','R','U','D'."""
    m = {"left":"L","right":"R","up":"U","down":"D"}
    out = []
    for t in text_dirs:
        key = t.strip().lower()
        if key in m:
            out.append(m[key])
    return out

# ------------- BUILD UNIFIED SAMPLES -------------

def base_header(rec_id: str, task_type: str, stage: int, difficulty: str, split: str, maze_str: str) -> Dict[str, Any]:
    lines, h, w = parse_maze(maze_str)
    return {
        "id": rec_id,
        "version": "v1.0",
        "task_type": task_type,
        "skills": ["ascii_maze","coord_system/top_left","dirs/U-D-L-R"],
        "skills_required": [],
        "curriculum_stage": stage,            # you can tune per task below
        "difficulty": difficulty,
        "meta": {
            "source": "existing",
            "generator": "unspecified",
            "tags": ["maze","ascii","converted"],
            "split": split
        },
        "maze": {
            "repr": maze_str,
            "height": h,
            "width": w,
            "walkable_chars": WALKABLE_CHARS
        },
        "io_format": "chat",
    }

def make_system_msg() -> Dict[str,str]:
    return {
        "role": "system",
        "content": "You are a maze assistant. Use 0-indexed rows/cols with top-left origin and the directions U,D,L,R."
    }

def sample_detect_start_end(base: Dict[str,Any], start: Tuple[int,int], end: Tuple[int,int], scratchpad: str) -> Dict[str,Any]:
    s = dict(base)  # shallow copy ok (we won’t mutate nested)
    s["task_type"] = "DETECT_START_END"
    s["curriculum_stage"] = 1
    s["messages"] = [
        make_system_msg(),
        {
            "role": "user",
            "content": "Given this ASCII maze, return JSON with the coordinates of start 'S' and end 'E' as {\"start\":{\"row\":r,\"col\":c},\"end\":{\"row\":r,\"col\":c}}.\n\n" + s["maze"]["repr"]
        }
    ]
    s["supervision"] = {
        "targets": {
            "start": {"row": start[0], "col": start[1]},
            "end": {"row": end[0], "col": end[1]},
        },
        "evaluation": {}
    }
    s["scratchpad"] = (scratchpad or "").strip()
    s["rationale_policy"] = SCRATCHPAD_POLICY
    return s

def sample_available_dirs(base: Dict[str,Any], grid: List[str], at: Tuple[int,int], avail_dirs: List[str], scratchpad: str) -> Dict[str,Any]:
    s = dict(base)
    s["task_type"] = "AVAILABLE_DIRECTIONS"
    s["curriculum_stage"] = 2
    s["skills_required"] = ["detect_start_end"]
    s["messages"] = [
        make_system_msg(),
        {
            "role": "user",
            "content": f"At coordinate {json.dumps({'row':at[0],'col':at[1]})}, list available directions (U,D,L,R) that move to walkable cells.\n\n{s['maze']['repr']}"
        }
    ]
    s["supervision"] = {
        "targets": {
            "at": {"row": at[0], "col": at[1]},
            "available_dirs": avail_dirs
        },
        "evaluation": {}
    }
    s["scratchpad"] = (scratchpad or "").strip()
    s["rationale_policy"] = SCRATCHPAD_POLICY
    return s

def sample_valid_move(base: Dict[str,Any], pos_from: Tuple[int,int], d: str, valid: bool, pos_to: Tuple[int,int] = None) -> Dict[str,Any]:
    s = dict(base)
    s["task_type"] = "VALID_MOVE"
    s["curriculum_stage"] = 2
    s["skills_required"] = ["available_directions"]
    s["messages"] = [
        make_system_msg(),
        {
            "role": "user",
            "content": f'From {json.dumps({"row":pos_from[0],"col":pos_from[1]})}, is moving "{d}" valid? Answer YES or NO and give the new coordinate if YES.'
        }
    ]
    targets = {
        "from": {"row": pos_from[0], "col": pos_from[1]},
        "dir": d,
        "valid": bool(valid),
    }
    if valid and pos_to is not None:
        targets["to"] = {"row": pos_to[0], "col": pos_to[1]}
    s["supervision"] = {"targets": targets, "evaluation": {}}
    s["scratchpad"] = ""
    s["rationale_policy"] = "omit_at_inference"
    return s

def sample_all_in_one(base: Dict[str,Any], start: Tuple[int,int], end: Tuple[int,int], runs: List[Dict[str,Any]], path_len: int, scratchpad: str) -> Dict[str,Any]:
    s = dict(base)
    s["task_type"] = "ALL_IN_ONE"
    s["skills_required"] = ["detect_start_end","available_directions","valid_move"]
    s["curriculum_stage"] = 4
    s["skills"] = s["skills"] + ["multi_step_planning"]
    s["messages"] = [
        make_system_msg(),
        {
            "role": "user",
            "content": "Return start/end, briefly note available directions you consider at each step, then output a valid path as runs.\n\n" + s["maze"]["repr"]
        }
    ]
    s["supervision"] = {
        "targets": {
            "start": {"row": start[0], "col": start[1]},
            "end": {"row": end[0], "col": end[1]},
            "solution_runs": runs,
            "path_length": path_len
        },
        "evaluation": {}
    }
    s["scratchpad"] = (scratchpad or "").strip()
    s["rationale_policy"] = SCRATCHPAD_POLICY
    return s

# ------------- MAIN CONVERSION -------------

def convert_record(rec: Dict[str,Any], idx: int, split: str) -> List[Dict[str,Any]]:
    """Convert one of your raw records into multiple unified samples."""
    out = []

    # Pull fields from your format
    maze_str = rec.get("maze", "")
    grid, h, w = parse_maze(maze_str)

    # start/end: prefer provided; else detect from grid
    ans = rec.get("answer", {}) or {}
    s_tuple = tuple(ans.get("start")) if "start" in ans else None
    e_tuple = tuple(ans.get("end")) if "end" in ans else None
    if s_tuple is None or e_tuple is None:
        auto_s, auto_e = find_S_E(grid)
        s_tuple = s_tuple or auto_s
        e_tuple = e_tuple or auto_e

    if s_tuple is None or e_tuple is None:
        # skip bad record
        return out

    # path directions (map text -> enum)
    raw_dirs_text = rec.get("directions", []) or []
    dirs_enum = map_text_dirs_to_enum(raw_dirs_text)
    runs = compress_runs(dirs_enum)
    difficulty = infer_difficulty(h, w, len(dirs_enum))

    # scratchpad (optional): we can stitch chain_of_thought fields if desired
    cot = rec.get("chain_of_thought", {}) or {}
    spad_lines = []
    if "start" in cot:
        spad_lines.append(cot["start"].strip())
    if "available_directions" in cot:
        spad_lines.append(cot["available_directions"].strip())
    scratchpad = ("\n".join(spad_lines)).strip()

    # Base header
    base = base_header(
        rec_id=f"rec{idx}",
        task_type="DETECT_START_END",
        stage=1,
        difficulty=difficulty,
        split=split,
        maze_str=maze_str,
    )

    # 1) DETECT_START_END
    if EMIT_TASKS["DETECT_START_END"]:
        out.append(
            sample_detect_start_end(
                base, s_tuple, e_tuple, scratchpad
            )
        )

    # 2) AVAILABLE_DIRECTIONS at the start location
    if EMIT_TASKS["AVAILABLE_DIRECTIONS"]:
        avail = available_dirs_at(grid, s_tuple[0], s_tuple[1])
        # If your record already has a single set in answer.available_directions, we can intersect
        provided = ans.get("available_directions", [])
        provided_enum = map_text_dirs_to_enum(provided)
        # Prefer computed (consistent), but keep provided if present and non-empty and consistent
        avail_dirs = avail if avail else provided_enum
        out.append(
            sample_available_dirs(
                base, grid, s_tuple, avail_dirs, scratchpad
            )
        )

    # 3) VALID_MOVE from the *true* path (positives) + a few negatives
    if EMIT_TASKS["VALID_MOVE"] and dirs_enum:
        cur = s_tuple
        # positives
        for step, d in enumerate(dirs_enum):
            nxt = apply_dir(cur, d)
            # validity check based on grid
            valid = is_walkable(char_at(grid, nxt[0], nxt[1]))
            out.append(sample_valid_move(base, cur, d, valid=valid, pos_to=nxt if valid else None))
            cur = nxt if valid else cur  # continue along if valid

        # negatives: sample some states from along the path and propose a blocked dir
        random_states = []
        cur = s_tuple
        random_states.append(cur)
        for d in dirs_enum[:max(1, len(dirs_enum)//3)]:
            cur = apply_dir(cur, d)
            random_states.append(cur)
        random.shuffle(random_states)
        tried = 0
        for pos in random_states:
            if tried >= NEG_MOVES_PER_RECORD:
                break
            r, c = pos
            legal = set(available_dirs_at(grid, r, c))
            illegal = [d for d in DIRS if d not in legal]
            if not illegal:
                continue
            d_bad = random.choice(illegal)
            nxt = apply_dir(pos, d_bad)
            out.append(sample_valid_move(base, pos, d_bad, valid=False, pos_to=None))
            tried += 1

    # 4) ALL_IN_ONE
    if EMIT_TASKS["ALL_IN_ONE"]:
        out.append(
            sample_all_in_one(
                base, s_tuple, e_tuple, runs, path_len=len(dirs_enum), scratchpad=scratchpad
            )
        )

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON or JSONL path")
    ap.add_argument("--out", dest="outp", required=True, help="Output JSONL path")
    ap.add_argument("--split", default="train", help="train|valid|test")
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.outp)
    rows = read_any_jsonlike(src)

    all_samples = []
    for i, rec in enumerate(rows):
        all_samples.extend(convert_record(rec, idx=i, split=args.split))

    # deterministic-ish order: group by id+task
    def order_key(s):
        return (s["id"], s["task_type"])
    all_samples.sort(key=order_key)

    write_jsonl(dst, all_samples)
    print(f"Wrote {len(all_samples)} samples to {dst}")

if __name__ == "__main__":
    main()
