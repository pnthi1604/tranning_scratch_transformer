import torch.nn as nn    

# const variable
XAVIER_UNIFORM = "xavier_uniform"

def _init_weights(
    module,
    init_type: str=None,
    mean: float=0.0,
    std: float=0.02
):
    if init_type is None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    elif init_type == XAVIER_UNIFORM:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

__all__ = [
    "_init_weights",
    "XAVIER_UNIFORM",
]