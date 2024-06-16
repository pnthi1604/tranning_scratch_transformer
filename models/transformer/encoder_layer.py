import torch
import torch.nn as nn
from .utils import (
    ACT_FN,
)
from .config import TransformerConfig
from .multihead_attn import MultiheadAttention

class EncoderLayer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = ACT_FN[config.activation_function]()
        self.activation_dropout = nn.Dropout((config.activation_dropout))
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor=None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states
    
__all__ = [
    "EncoderLayer",
]