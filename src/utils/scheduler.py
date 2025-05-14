from dataclasses import dataclass


@dataclass
class LearningRateConfig:
    start: float = 255 * 3e-3
    end: float = 255 * 3e-4


class LearningRateScheduler:
    def __init__(self, lr: LearningRateConfig, n_iter):
        self.lr = lr
        self.n_iter = n_iter
    
    def get_lr(self, iter: int):
        # simple linear interpolation for now
        lr = self.lr.start + (self.lr.end - self.lr.start) * iter / self.n_iter
        return lr
