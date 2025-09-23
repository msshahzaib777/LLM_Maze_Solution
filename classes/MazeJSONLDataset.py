import json

class MazeJSONLDataset:
    """
    Expects a JSONL file with rows like:
    {
      "maze": "#########\\n#   S  E\\n#########",
      "prompt": "Identify the start location and its available directions in this maze.",
      "answer": {"start":[r,c], "available_directions":["left","up"] }
      // optional: "chain_of_thought", "solved_maze"
    }
    Emits dicts with 'input_ids' and 'labels' (prompt masked with -100).
    """
    def __init__(self,
                 jsonl_path: str,
                 tokenizer,
                 max_seq_len: int = 256,
                 target_mode: str = "start_and_dirs"  # or "start_only"
                 ):
        self.items: List[Dict[str, Any]] = []
        self.tok = tokenizer
        self.max_len = max_seq_len
        self.pad_id = tokenizer.pad_id if hasattr(tokenizer, "pad_id") and tokenizer.pad_id is not None else 0
        self.target_mode = target_mode

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                maze = ex["maze"]
                user_prompt = ex.get("prompt", "Identify the start location and its available directions in this maze.")
                # Ground truth:
                ans = ex.get("answer", {})
                start = ans.get("start", None)
                dirs  = ans.get("available_directions", None)

                # Build strings
                prompt_text = build_prompt(maze, user_prompt)

                if self.target_mode == "start_only":
                    target_obj = {"start": start}
                    target_text = json.dumps(target_obj, ensure_ascii=False)
                else:
                    target_text = build_target(start, dirs)

                # Tokenize full sequence = [prompt][assistant target]
                prompt_ids = self.tok.encode(prompt_text)
                target_ids = self.tok.encode(target_text)

                input_ids = prompt_ids + target_ids
                input_ids = clamp_and_pad(input_ids, self.max_len, self.pad_id)

                # Labels: mask prompt part (set to -100), keep target tokens
                labels = [-100] * min(len(prompt_ids), self.max_len)
                tail = self.max_len - len(labels)
                if tail > 0:
                    # the tail corresponds to (part of) target_ids (padded to max_len)
                    tgt_tail = clamp_and_pad(target_ids, tail, self.pad_id)
                    # masked positions for padding should also be -100 (so they don’t contribute to loss)
                    labels += [tid if tid != self.pad_id else -100 for tid in tgt_tail]
                else:
                    labels = labels[:self.max_len]

                self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
