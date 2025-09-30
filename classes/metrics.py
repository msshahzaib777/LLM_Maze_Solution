import math
from typing import Dict, List, Optional, Tuple, Union

class SimpleMetrics:
    def __init__(self):
        self.it = 0
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses:   List[Tuple[int, float]] = []
        self.train_info = []
    def on_train_step_end(self, it: int, loss: float):
        self.train_losses.append((it, float(loss)))
    def on_eval_end(self, it: int, loss: float):
        self.val_losses.append((it, float(loss)))
    def on_val_loss_report(self, val_info):
        self.val_losses.append((val_info["iteration"], float(val_info["val_loss"])))
    def on_train_loss_report(self, train_info):
        self.train_losses.append((train_info["iteration"], float(train_info["train_loss"])))
        self.train_info.append(train_info)

def cosine_with_warmup(step, base_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / float(warmup_steps)
    p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

class LRSchedulerCallback(SimpleMetrics):
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps=0, min_lr_ratio=0.1, log_every=50):
        super().__init__()
        self.opt = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.log_every = log_every
        self.lr_track = []

    def on_step_end(self, step, loss, is_eval=False, **kwargs):
        super().on_step_end(step, loss, is_eval=is_eval, **kwargs)
        if not is_eval:
            new_lr = cosine_with_warmup(
                step=step,
                base_lr=self.base_lr,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps,
                min_lr_ratio=self.min_lr_ratio,
            )
            self.opt.learning_rate = new_lr
            self.lr_track.append((step, float(new_lr)))
            if step % self.log_every == 0:
                print(f"[{step}] lr -> {float(new_lr):.6e}")
