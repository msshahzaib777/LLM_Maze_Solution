import json, random
from pathlib import Path

# --- Dataset helpers ---------------------------------------------------------

def _read_json_any(path: Path):
    """Return a list of dict records from JSON, JSON array, or JSONL."""
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # Try regular JSON (object or array)
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    # Fallback: JSONL
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def _load_split(dir_or_file: Union[str, Path]):
    """Load *.json / *.jsonl from a directory OR a single file path."""
    p = Path(dir_or_file)
    files = []
    if p.is_dir():
        files += sorted(p.glob("*.json"))
        files += sorted(p.glob("*.jsonl"))
    else:
        files = [p]
    rows = []
    for fp in files:
        rows.extend(_read_json_any(fp))
    return rows

def _to_messages(record: Dict):
    """Normalize different schemas into chat messages."""
    if isinstance(record.get("messages"), list):
        return record["messages"]

    # Your schema: prompt/answer
    if "prompt" in record and "answer" in record:
        return [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["answer"]},
        ]

    # Common alt schema
    if "input" in record and "output" in record:
        return [
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["output"]},
        ]

    # Fallback: dump the record to the user side
    return [{"role": "user", "content": json.dumps(record, ensure_ascii=False)}]

def _build_texts(rows, tokenizer: PreTrainedTokenizer):
    """Apply chat template; append EOS so packed batches separate cleanly."""
    texts = []
    for r in rows:
        msgs = _to_messages(r)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        if getattr(tokenizer, "eos_token", None) and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        texts.append(text)
    return texts

class TextDataset:
    """Minimal dataset the MLX trainer can consume (returns raw strings)."""
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

# --- Load your splits --------------------------------------------------------

# Point these to your files OR directories
train_path = "/path/to/train"  # e.g., "/data/maze/train.jsonl" or "/data/maze/train_dir/"
val_path   = "/path/to/val"
test_path  = "/path/to/test"   # optional

train_rows = _load_split(train_path)
val_rows   = _load_split(val_path)

random.shuffle(train_rows)  # good practice for SFT

train_texts = _build_texts(train_rows, tokenizer)
val_texts   = _build_texts(val_rows, tokenizer)

train_set = TextDataset(train_texts)
val_set   = TextDataset(val_texts)
