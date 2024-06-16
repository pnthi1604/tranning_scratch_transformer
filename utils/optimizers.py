import torch

def get_AdamW(
    model,
    lr: float = 0.001,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0,
):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

def get_RAdam(
    model,
    lr: float=0.001,
    betas: tuple=(0.9, 0.999),
    eps: float=1e-8,
    weight_decay: float=0,
):
    return torch.optim.RAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

# const optimizer
ADAMW = "AdamW"
RADAM = "RAdam"

GET_OPTIMIZER = {
    ADAMW: get_AdamW,
    RADAM: get_RAdam,
}

__all__ = ["GET_OPTIMIZER", "ADAMW", "RADAM"]