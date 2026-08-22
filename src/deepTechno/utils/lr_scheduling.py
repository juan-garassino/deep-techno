import math


class LrStepTracker:
    """Warmup LR schedule from 'Attention is All You Need' (Vaswani et al. 2017)."""

    def __init__(self, model_dim=512, warmup_steps=4000, init_steps=0):
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        self.init_steps = init_steps
        self.invsqrt_dim = 1 / math.sqrt(model_dim)
        self.invsqrt_warmup = 1 / (warmup_steps * math.sqrt(warmup_steps))

    def step(self, step):
        step += self.init_steps
        if step <= self.warmup_steps:
            return self.invsqrt_dim * self.invsqrt_warmup * step
        return self.invsqrt_dim / math.sqrt(step)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
