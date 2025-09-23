from typing import Dict, List, Optional, Tuple, Union

class SimpleMetrics:
    def __init__(self):
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses:   List[Tuple[int, float]] = []
    def on_train_step_end(self, it: int, loss: float):
        self.train_losses.append((it, float(loss)))
    def on_eval_end(self, it: int, loss: float):
        self.val_losses.append((it, float(loss)))