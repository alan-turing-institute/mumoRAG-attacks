class LearningRateScheduler:
    def __init__(self, lr_start, lr_end, n_iter):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.n_iter = n_iter
    
    def get_lr(self, iter: int):
        # simple linear interpolation for now
        lr = self.lr_start + (self.lr_end - self.lr_start) * iter / self.n_iter
        return lr