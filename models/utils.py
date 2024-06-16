import torch
from .seq2seq import (
    TransformerSeq2seq,
    TransformerSeq2seqConfig,
)

# const variables
SEQ2SEQ_TRANSFORMER = "transformer_seq2seq"

# models dict
models = {
    SEQ2SEQ_TRANSFORMER: {
        "config": TransformerSeq2seqConfig,
        "model": TransformerSeq2seq,
    },
}

def get_model(
    model_name: str,
    **kwargs,
):
    return models[model_name]["model"](
        config=models[model_name]["config"](
            **kwargs,
        ),
    )

def show_layer_un_freeze(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def freeze_model(model, modules=[]):
    for module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = False
    return model

def un_freeze_model(model, modules=[]):
    for module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = True
    return model

# load model state dict
def load_model(checkpoint, model):
    if torch.cuda.is_available():
        state = torch.load(checkpoint)
    else:
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state["model_state_dict"])
    return model

__all__ = [
    "get_model",
    "show_layer_un_freeze",
    "freeze_model",
    "un_freeze_model",
    "load_model",
    "SEQ2SEQ_TRANSFORMER",
]