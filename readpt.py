import os
import sys
import argparse
from typing import Any, Dict, List

import torch


def _fmt_float(x: Any, ndigits: int = 6):
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def summarize_optimizer_state(opt_state: Dict[str, Any]) -> List[str]:
    lines = []
    try:
        # Common keys present in AdamW state dict: 'state', 'param_groups'
        if "param_groups" in opt_state:
            lines.append(f"Optimizer: param_groups = {len(opt_state['param_groups'])}")
            # Show LR(s) and betas from groups
            for i, g in enumerate(opt_state["param_groups"]):
                lr = g.get("lr", None)
                betas = g.get("betas", None)
                wd = g.get("weight_decay", None)
                lines.append(
                    f"  Group {i}: lr={lr:.6e}" if isinstance(lr, (int, float)) else f"  Group {i}: lr={lr}"
                )
                if betas is not None:
                    lines.append(f"           betas={betas}")
                if wd is not None:
                    lines.append(f"           weight_decay={wd}")
        if "state" in opt_state:
            lines.append(f"Optimizer: parameter_states = {len(opt_state['state'])}")
    except Exception as e:
        lines.append(f"(Could not summarize optimizer state: {e})")
    return lines


def summarize_scheduler_state(sched_state: Dict[str, Any]) -> List[str]:
    lines = []
    try:
        # LambdaLR / Cosine schedule often has 'last_epoch'
        last_epoch = sched_state.get("last_epoch", None)
        if last_epoch is not None:
            lines.append(f"Scheduler: last_epoch = {last_epoch}")
        else:
            lines.append("Scheduler: (no 'last_epoch' key found)")
    except Exception as e:
        lines.append(f"(Could not summarize scheduler state: {e})")
    return lines


def dump_training_state(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("TRAINING STATE SUMMARY")
    lines.append("=" * 80)

    step = state.get("step", None)
    lines.append(f"Step: {step}")

    # Optimizer summary
    opt_state = state.get("optimizer_state_dict", {})
    lines.append("-" * 80)
    lines.append("Optimizer State")
    lines.extend(summarize_optimizer_state(opt_state))

    # Scheduler summary
    sched_state = state.get("scheduler_state_dict", {})
    lines.append("-" * 80)
    lines.append("Scheduler State")
    lines.extend(summarize_scheduler_state(sched_state))

    # Loss history
    loss_history = state.get("loss_history", [])
    lines.append("-" * 80)
    lines.append(f"Loss History: {len(loss_history)} entries")
    if loss_history:
        lines.append(f"{'Index':>5}  {'Step':>10}  {'Loss':>12}  {'LR':>12}")
        for i, item in enumerate(loss_history):
            s = item.get("step", "")
            l = _fmt_float(item.get("loss", ""))
            lr = item.get("lr", "")
            lr_str = f"{lr:.6e}" if isinstance(lr, (int, float)) else str(lr)
            lines.append(f"{i:>5}  {s:>10}  {l:>12}  {lr_str:>12}")

        # Quick summary
        first = loss_history[0]
        last = loss_history[-1]
        try:
            imp = float(first.get("loss", 0.0)) - float(last.get("loss", 0.0))
            lines.append("")
            lines.append(
                f"Initial loss: {_fmt_float(first.get('loss', ''))} | "
                f"Final loss: {_fmt_float(last.get('loss', ''))} | "
                f"Improvement: {_fmt_float(imp)}"
            )
        except Exception:
            pass

    # Eval loss history
    eval_loss_history = state.get("eval_loss_history", [])
    lines.append("-" * 80)
    lines.append(f"Eval Loss History: {len(eval_loss_history)} entries")
    if eval_loss_history:
        lines.append(f"{'Index':>5}  {'Step':>10}  {'Val Loss':>12}")
        for i, item in enumerate(eval_loss_history):
            s = item.get("step", "")
            vl = _fmt_float(item.get("val_loss", ""))
            lines.append(f"{i:>5}  {s:>10}  {vl:>12}")

    # Fallback: list any unexpected top-level keys
    expected = {
        "step",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "loss_history",
        "eval_loss_history",
    }
    unknown = [k for k in state.keys() if k not in expected]
    if unknown:
        lines.append("-" * 80)
        lines.append("Other top-level keys found:")
        for k in unknown:
            v = state[k]
            vtype = type(v).__name__
            try:
                shape = getattr(v, "shape", None)
                if shape is not None:
                    lines.append(f"  {k}: {vtype} shape={tuple(shape)}")
                else:
                    lines.append(f"  {k}: {vtype}")
            except Exception:
                lines.append(f"  {k}: {vtype}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Dump a .pt training state into a .txt report.")
    ap.add_argument("input", help="Path to .pt file OR a directory containing training_state.pt")
    ap.add_argument(
        "-o",
        "--output",
        help="Path to output .txt file (default: same directory, name based on input).",
        default=None,
    )
    ap.add_argument(
        "--map-location",
        default="cpu",
        help="torch.load map_location (default: cpu). Examples: cpu, cuda, mps",
    )
    args = ap.parse_args()

    # If a directory is passed, assume training_state.pt inside it
    in_path = args.input
    if os.path.isdir(in_path):
        candidate = os.path.join(in_path, "training_state.pt")
        if os.path.exists(candidate):
            in_path = candidate
        else:
            print(f"Directory provided, but no 'training_state.pt' found in: {args.input}", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(in_path):
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Load state
    try:
        state = torch.load(in_path, map_location=args.map_location)
    except Exception as e:
        print(f"Failed to load {in_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Some checkpoints are wrapped; handle a generic dict or nested
    if isinstance(state, dict) and "state" in state and isinstance(state["state"], dict):
        # e.g., {'state': {...}} — unwrap if it looks like the saved structure
        inner = state["state"]
        # If this inner dict has the expected keys, use it
        if any(k in inner for k in ("step", "loss_history", "optimizer_state_dict")):
            state = inner

    if not isinstance(state, dict):
        print("Loaded object is not a dict. This script expects a state dict saved like in your training code.", file=sys.stderr)
        sys.exit(1)

    report = dump_training_state(state)

    # Output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_dir = os.path.dirname(os.path.abspath(in_path))
        out_path = os.path.join(out_dir, f"{base}_summary.txt")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote summary to: {out_path}")


if __name__ == "__main__":
    main()

