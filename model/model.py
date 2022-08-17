"""
@File        :  model.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/16
@Version     :  1.0
@Description :  Bert with variable length of `[CLS]`
"""
import copy
import random
from typing import Optional, Tuple, Union

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel, BertConfig, BartConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bart.modeling_bart import BartDecoder
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertEncoder


class LogLMConfig(BertConfig):
    model_type = "bert"

    def __init__(
            self,
            vocab_size=30522,
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
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            shared_non_casual_decoder=False,
            shared_casual_decoder=True,
            num_non_casual_decoder_layers=2,
            num_casual_decoder_layers=2,
            mask_cls_prob=0.2,
            cls_noise_std=0.002,
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

        self.shared_non_casual_decoder = shared_non_casual_decoder
        self.shared_casual_decoder = shared_casual_decoder
        self.num_non_casual_decoder_layers = num_non_casual_decoder_layers
        self.num_casual_decoder_layers = num_casual_decoder_layers
        self.mask_cls_prob = mask_cls_prob
        self.cls_noise_std = cls_noise_std

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
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_labels=2,
            pad_token_id=0,
            bos_token_id=14,  # [SEP]
            eos_token_id=14,
            decoder_start_token_id=14,
            forced_eos_token_id=14,
        )
        config.hidden_act = self.hidden_act
        config.layer_norm_eps = self.layer_norm_eps
        return config


@dataclass
class LogLMForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    sop_loss: Optional[torch.FloatTensor] = None
    dec_mlm_loss: Optional[torch.FloatTensor] = None

    prediction_logits: Optional[torch.FloatTensor] = None
    dec_prediction_logits: Optional[torch.FloatTensor] = None
    seq_relationship_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LogLMPreTrainedModel(BertPreTrainedModel):
    config_class = LogLMConfig

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def _reorder_cache(self, past, beam_idx):
        pass


class NonCausalDecoder(nn.Module):
    def __init__(self, config: LogLMConfig, shared_decoder: BertEncoder = None,
                 shared_lm_predictions: BertLMPredictionHead = None):
        super().__init__()
        self.config = config.for_non_casual_decoder()
        self.decoder = BertEncoder(self.config) if shared_decoder is None else shared_decoder
        self.lm_predictions = BertLMPredictionHead(self.config) if shared_lm_predictions is None \
            else shared_lm_predictions

    def forward(self, hidden_states, attention_mask, head_mask=None, labels=None):
        batch_size, seq_length, _ = hidden_states.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)

        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        outputs = self.decoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        logits = self.lm_predictions(outputs.last_hidden_state)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return loss, logits


class CausalDecoder(nn.Module):
    def __init__(self, config: LogLMConfig, embeddings: nn.Embedding = None,
                 shared_lm_predictions: BertLMPredictionHead = None):
        super().__init__()
        self.config = config.for_casual_decoder()
        self.decoder = BartDecoder(self.config, embed_tokens=embeddings)
        self.lm_predictions = BertLMPredictionHead(self.config) if shared_lm_predictions is None \
            else shared_lm_predictions

    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, labels=None,
                past_key_values=None):
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = self.lm_predictions(outputs.last_hidden_state)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return loss, logits


class LogLMForPreTraining(LogLMPreTrainedModel):
    def __init__(self, config: LogLMConfig):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh())

        self.lm_predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

        if config.shared_non_casual_decoder:
            self.non_casual_decoder = NonCausalDecoder(
                config, shared_decoder=self.bert.encoder, shared_lm_predictions=self.lm_predictions
            )
        elif config.num_non_casual_decoder_layers > 0:
            self.non_casual_decoder = NonCausalDecoder(config)
        else:
            self.non_casual_decoder = None

        if config.shared_casual_decoder:
            self.casual_decoder = CausalDecoder(
                config, embeddings=self.bert.embeddings.word_embeddings, shared_lm_predictions=self.lm_predictions
            )
        elif config.num_casual_decoder_layers > 0:
            self.casual_decoder = CausalDecoder(config)
        else:
            self.casual_decoder = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            cls_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            sentence_pair_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mask_token_id=None,
    ):
        assert mask_token_id is not None, f"Require `mask_token_id`"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        cal_mlm_loss = (labels is not None)
        cal_nsp_loss = (sentence_pair_label is not None)

        if (labels is not None) and (cls_type_ids is None):
            cls_type_ids = (1 <= input_ids) & (input_ids <= 11)
            cls_type_ids[:, 0] = True

        # First pass: nsp_loss + mlm_loss
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state

        loss = mlm_loss = sop_loss = dec_mlm_loss = None
        prediction_logits = dec_prediction_logits = seq_relationship_logits = None

        loss_fct = CrossEntropyLoss(ignore_index=-100)

        # Mask some positions (from back to front)
        masked_cls_mask = None

        # Prepare cls mask
        if cal_mlm_loss or cal_nsp_loss:
            if self.training and random.random() < self.config.mask_cls_prob:
                cls_len = cls_type_ids.sum(dim=-1)
                masked_cls_mask = torch.zeros_like(cls_type_ids, dtype=torch.float)
                for idx, length in enumerate(cls_len):
                    reserved_length = random.randint(1, length)
                    masked_cls_mask[idx, :reserved_length] = 1
            else:
                masked_cls_mask = cls_type_ids.float()
            masked_cls_mask = masked_cls_mask.unsqueeze(-1)

        if cal_mlm_loss:
            prediction_logits = self.lm_predictions(last_hidden_state)
            labels_ = labels.clone()
            labels_[input_ids != mask_token_id] = -100
            mlm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels_.view(-1))
            loss = mlm_loss

        if cal_nsp_loss:
            masked_cls_len = masked_cls_mask.sum(dim=1)  # [B, 1]
            cls_hiddens = (last_hidden_state * masked_cls_mask).sum(dim=1)  # [B, D]
            cls_hiddens = cls_hiddens / masked_cls_len
            seq_relationship_logits = self.seq_relationship(cls_hiddens)
            sop_loss = loss_fct(seq_relationship_logits.view(-1, 2), sentence_pair_label.view(-1))
            loss = sop_loss if (loss is None) else loss + sop_loss

        # Second pass: mlm_loss
        if cal_mlm_loss:
            if self.non_casual_decoder is not None:
                # dummy_input_ids = torch.zeros_like(cls_type_ids, dtype=torch.long).fill_(self.config.pad_token_id)
                # dummy_input_ids[torch.where(attention_mask)] = mask_token_id
                dummy_input_ids = attention_mask * mask_token_id + (1 - attention_mask) * self.config.pad_token_id
                dummy_embeds = self.bert.embeddings.forward(dummy_input_ids, token_type_ids, position_ids)  # [B, L, D]

                inputs_embeds = (
                        self.cls_proj(last_hidden_state) * masked_cls_mask + dummy_embeds * (1 - masked_cls_mask)
                )

                # Add dropout noise on cls
                inputs_embeds = nn.functional.dropout(inputs_embeds, self.config.hidden_dropout_prob, self.training)

                dec_mlm_loss, dec_prediction_logits = self.non_casual_decoder.forward(
                    hidden_states=inputs_embeds,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    labels=labels,
                )
                loss = dec_mlm_loss if (loss is None) else loss + dec_mlm_loss

            if self.casual_decoder is not None:
                # dec_lm_loss, dec_lm_prediction_logits = self.casual_decoder.forward(
                #     input_ids=None,
                #     attention_mask=None,
                #     encoder_hidden_states=None,
                #     encoder_attention_mask=None,
                #     labels=labels
                # )
                # loss = dec_lm_loss if (loss is None) else loss + dec_lm_loss
                pass  # TODO: Implement

        if not return_dict:
            output = (prediction_logits,) + outputs[2:]
            return ((loss, mlm_loss, sop_loss, dec_mlm_loss) + output) if loss is not None else output

        return LogLMForPreTrainingOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            sop_loss=sop_loss,
            dec_mlm_loss=dec_mlm_loss,
            prediction_logits=prediction_logits,
            dec_prediction_logits=dec_prediction_logits,
            seq_relationship_logits=seq_relationship_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LogLMForSequenceClassification(LogLMPreTrainedModel):
    def __init__(self, config: LogLMConfig, pooling_all=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout if (config.classifier_dropout is not None)
                       else config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.pooling_all = pooling_all  # Pooling all hidden states

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            cls_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.pooling_all:
            div_length = attention_mask.sum(dim=-1).unsqueeze(-1)
            pooled_output = (outputs.last_hidden_state * attention_mask.float().unsqueeze(-1)).sum(dim=1)  # [B, D]
        else:
            div_length = cls_type_ids.sum(dim=-1).unsqueeze(-1)
            pooled_output = (outputs.last_hidden_state * cls_type_ids.float().unsqueeze(-1)).sum(dim=1)  # [B, D]

        pooled_output = pooled_output / div_length
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
