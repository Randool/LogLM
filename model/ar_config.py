"""
@File        :  config.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/23
@Version     :  1.0
@Description :  None
"""
import warnings
from typing import Union, List

from transformers import PretrainedConfig, LongformerConfig


class LogLMConfig(PretrainedConfig):
    model_type = "loglm"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
            self,
            vocab_size=20000,
            max_position_embeddings=1024,
            encoder_layers=12,
            encoder_ffn_dim=3072,
            encoder_attention_heads=16,
            decoder_layers=12,
            decoder_ffn_dim=3072,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            activation_function="gelu",
            d_model=768,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            classifier_dropout=0.0,
            scale_embedding=False,
            use_cache=True,
            num_labels=3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=3,
            is_encoder_decoder=True,
            decoder_start_token_id=3,
            forced_eos_token_id=3,
            use_longformer=False,
            log_cross_attn=True,
            attention_window: Union[List[int], int] = 512,
            **kwargs
    ):
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        # For Longformer
        self.use_longformer = use_longformer
        self.log_cross_attn = log_cross_attn
        self.attention_window = attention_window
        self.sep_token_id = eos_token_id

        # ensure backward compatibility for BART CNN models
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )

    def as_longformer_config(self) -> LongformerConfig:
        return LongformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.d_model,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.encoder_attention_heads,
            intermediate_size=self.encoder_ffn_dim,
            hidden_act=self.activation_function,
            hidden_dropout_prob=self.dropout,
            max_position_embeddings=4098,
            type_vocab_size=2,
            initializer_range=self.init_std,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            sep_token_id=self.sep_token_id,
            attention_window=self.attention_window,
        )
