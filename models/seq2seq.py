import torch
import torch.nn as nn

from .transformer import (
    Encoder,
    Decoder,
    Embeds,
    EncoderOut,
    DecoderOut,
    TransformerConfig,
)
from .transformer.utils.init_weights import (
    _init_weights,
    XAVIER_UNIFORM,
)

class TransformerSeq2seqConfig(TransformerConfig):
    def __init__(
        self,
        pad_token_id: int = 2,
        shared_embed: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer_config = TransformerConfig(**kwargs)
        self.pad_token_id = pad_token_id
        self.shared_embed = shared_embed

class TransformerSeq2seq(nn.Module):
    def __init__(
        self,
        config: TransformerSeq2seqConfig,
    ):
        super().__init__()
        self.config = config

        # Src embeds
        self.inputs_embeds = Embeds(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
            init_type=config.init_type,
            init_std=config.init_std,
            init_mean=config.init_mean,
        )

        # Tgt embeds
        self.decoder_inputs_embeds = Embeds(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
            init_type=config.init_type,
            init_std=config.init_std,
            init_mean=config.init_mean,
        )

        # Shared embeds
        if config.shared_embed:
            self.decoder_inputs_embeds.set_embed_tokens(self.inputs_embeds.embed_tokens)

        # Encoder
        self.encoder = Encoder(
            config=config.transformer_config,
        )

        # Decoder
        self.decoder = Decoder(
            config=config.transformer_config,
        )

        # Linear layer
        self.out = nn.Linear(
            in_features=config.d_model,
            out_features=config.tgt_vocab_size,
        )

        # Initialize weights
        modules = [self.out]
        for module in modules:
            _init_weights(
                module=module,
                init_type=config.init_type,
                init_std=config.init_std,
                mean=config.init_mean,
            )
        
    def forward(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        label: torch.Tensor=None,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        # encoder
        if inputs_embeds is not None:
            encoder_hidden_states = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    inputs_embeds=inputs_embeds,
                ),
                attention_mask=attention_mask,
            )
        else:
            encoder_hidden_states = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    input_ids=input_ids,
                ),
                attention_mask=attention_mask,
            )
        # decoder
        decoder_hidden_states = self.decoder(
            inputs_embeds=self.decoder_inputs_embeds(decoder_input_ids),
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        # out
        logits = self.out(decoder_hidden_states)

        if label is not None:
            if self.config.pad_token_id is not None:
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.config.pad_token_id,
                    label_smoothing=0.01,
                )
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            loss = loss_fn(logits.view(-1, self.config.tgt_vocab_size), label.view(-1))
            return logits, loss
        else:
            return logits
        
    def get_encoder_out(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        if inputs_embeds is not None:
            encoder_out = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    inputs_embeds=inputs_embeds,
                ),
                attention_mask=attention_mask,
            )
        else:
            encoder_out = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    input_ids=input_ids,
                ),
                attention_mask=attention_mask,
            )

        return EncoderOut(
            logits=encoder_out,
        )
    
    def get_decoder_out(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        decoder_out = self.decoder(
            inputs_embeds=self.decoder_inputs_embeds(input_ids),
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        return DecoderOut(
            logits=decoder_out,
        )
    
__all__ = [
    "TransformerSeq2seqConfig",
    "TransformerSeq2seq",
]