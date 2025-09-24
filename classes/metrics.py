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