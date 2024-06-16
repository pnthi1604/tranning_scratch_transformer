import torch
import torch.nn as nn
from .utils import (
    create_encoder_atn_mask,
)
from .utils.init_weights import (
    _init_weights,
)
from .config import TransformerConfig
from .encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.apply(lambda module: _init_weights(
            module=module,
            init_type=config.init_type,
            init_mean=config.init_mean,
            std=config.init_std,
        ))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        hidden_states = inputs_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = create_encoder_atn_mask(
                attention_mask=attention_mask,
            )

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs

        return hidden_states
    
__all__ = [
    "Encoder",
]