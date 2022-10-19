"""
@File        :  nar_config.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/10/13
@Version     :  1.0
@Description :  None
"""
import copy

from transformers import BertConfig, BartConfig


class LogLMNARConfig(BertConfig):
    model_type = "bert"

    def __init__(
            self,
            vocab_size=20000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=3,
            mask_token_id=4,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            num_non_casual_decoder_layers=2,
            num_casual_decoder_layers=2,
            mask_cls_prob=0.2,
            **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            **kwargs
        )

        self.num_non_casual_decoder_layers = num_non_casual_decoder_layers
        self.num_casual_decoder_layers = num_casual_decoder_layers
        self.mask_cls_prob = mask_cls_prob

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.decoder_start_token_id = eos_token_id

        self.is_encoder_decoder = self.num_casual_decoder_layers > 0

    def for_non_casual_decoder(self):
        config = copy.deepcopy(self)
        config.num_hidden_layers = config.num_non_casual_decoder_layers
        return config

    def for_casual_decoder(self):
        config = BartConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            encoder_layers=self.num_hidden_layers,
            encoder_ffn_dim=self.intermediate_size,
            encoder_attention_heads=self.num_attention_heads,
            decoder_layers=self.num_casual_decoder_layers,
            decoder_ffn_dim=self.intermediate_size,
            activation_function=self.hidden_act,
            d_model=self.hidden_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_labels=2,
            pad_token_id=0,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            forced_eos_token_id=self.forced_eos_token_id,
        )
        config.hidden_act = self.hidden_act
        config.layer_norm_eps = self.layer_norm_eps
        return config
