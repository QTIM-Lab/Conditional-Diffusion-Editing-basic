
def cosine_with_warmup(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1. + torch.cos(torch.tensor(progress * 3.1415926535))))
    return lr_lambda
