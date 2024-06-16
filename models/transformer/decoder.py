import torch
import torch.nn as nn
from .utils.mask import (
    create_decoder_atn_mask,
    create_encoder_atn_mask,
)
from .utils.init_weights import (
    _init_weights,
)
from .config import TransformerConfig
from .decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.apply(lambda module: _init_weights(
            module=module,
            init_type=config.init_type,
            std=config.init_std,
            mean=config.init_mean,
        ))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor=None,
        encoder_attention_mask: torch.Tensor=None,
    ):
        hidden_states = inputs_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = create_decoder_atn_mask(
                attention_mask=attention_mask,
                tgt_len=inputs_embeds.shape[1],
            )
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = create_encoder_atn_mask(
                attention_mask=encoder_attention_mask,
            )

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = layer_outputs

        return hidden_states
    
__all__ = [
    "Decoder",
]