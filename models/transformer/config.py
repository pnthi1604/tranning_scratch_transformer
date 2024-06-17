import torch

class TransformerConfig:
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        encoder_ffn_dim: int = 2048,
        activation_function: str = "relu",
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        max_position_embeddings: int = 1024,
        init_type: str = None,
        init_std: float = 0.02,
        init_mean: float = 0.0,
        label_smoothing: float = 0.01,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.init_type = init_type
        self.init_std = init_std
        self.init_mean = init_mean
        self.label_smoothing = label_smoothing

__all__ = [
    "TransformerConfig",
]