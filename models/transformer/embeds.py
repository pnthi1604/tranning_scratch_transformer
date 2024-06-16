import torch
import torch.nn as nn
from .utils.init_weights import (
    XAVIER_UNIFORM,
    _init_weights,
)

class Embeds(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        max_position_embeddings: int=1024,
        shared: bool = False,
        embed_scale: float=1.0,
        embed_tokens: nn.Embedding=None,
        init_type=XAVIER_UNIFORM,
        init_std: float=0.02,
        init_mean: float=0.0,
    ):
        super().__init__()

        self.embed_scale = embed_scale
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
            )
        self.embed_positions = nn.Embedding(
            max_position_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.register_buffer(
            "pos_ids",
            torch.arange(0, max_position_embeddings)
        )
        if shared:
            self.embed_positions.weight = self.embed_tokens.weight

        self.apply(lambda module: _init_weights(
            module=module,
            init_type=init_type,
            mean=init_mean,
            std=init_std,
        ))
    
    def set_embed_tokens(self, embed_tokens: nn.Embedding):
        self.embed_tokens = embed_tokens

    def forward(
        self, 
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
        else:
            bsz, seq_len, d_model = inputs_embeds.size()
        pos_ids = self.pos_ids[:seq_len]
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds * self.embed_scale + self.embed_positions(pos_ids)
    
__all__ = [
    "Embeds",
]