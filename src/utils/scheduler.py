class LearningRateScheduler():
    def __init__(self, start_lr, end_lr, n_iter):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.n_iter = n_iter
    
    def get_lr(self, iter: int):
        # simple linear interpolation for now
        lr = self.start_lr + (self.end_lr - self.start_lr) * iter / self.n_iter
        return lr