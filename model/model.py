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
from typing import Optional, Tuple, Union, Dict

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel, BertConfig, BartConfig
from transformers.file_utils import ModelOutput
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bart.modeling_bart import BartDecoder, shift_tokens_right
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertEncoder
from transformers.utils import logging

logger = logging.get_logger(__name__)


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
            bos_token_id=1,
            pad_token_id=0,
            eos_token_id=14,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            shared_non_casual_decoder=False,
            shared_casual_decoder=False,
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

        self.shared_non_casual_decoder = shared_non_casual_decoder
        self.shared_casual_decoder = shared_casual_decoder
        self.num_non_casual_decoder_layers = num_non_casual_decoder_layers
        self.num_casual_decoder_layers = num_casual_decoder_layers
        self.mask_cls_prob = mask_cls_prob

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = eos_token_id

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
    ndec_mlm_loss: Optional[torch.FloatTensor] = None
    dec_mlm_loss: Optional[torch.FloatTensor] = None

    prediction_logits: Optional[torch.FloatTensor] = None
    ndec_prediction_logits: Optional[torch.FloatTensor] = None
    dec_prediction_logits: Optional[torch.FloatTensor] = None
    seq_relationship_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LogLMPreTrainedModel(BertPreTrainedModel, GenerationMixin):
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

    def forward(self, input_ids, attention_mask, encoder_hidden_states,
                encoder_attention_mask, labels=None, past_key_values=None):
        outputs = self.decoder.forward(
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

    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        device = self.device
        inputs = {
            'input_ids': torch.tensor(
                [[1, 2, 3, 4, 54, 54, 54, 54, 14, 55, 55, 55, 55, 55, 55, 14],
                 [1, 2, 3, 4, 77, 77, 77, 14, 78, 78, 78, 78, 14, 0, 0, 0]], dtype=torch.long, device=device),
            'token_type_ids': torch.tensor(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long, device=device),
            'cls_type_ids': torch.tensor(
                [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long, device=device),
            'attention_mask': torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long, device=device),
            'mask_token_id': 15,
        }
        inputs["labels"] = inputs["input_ids"]
        return inputs

    def get_encoder(self):
        return self.bert

    def get_decoder(self):
        return self.casual_decoder

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

        loss = mlm_loss = sop_loss = ndec_mlm_loss = dec_mlm_loss = None
        ndec_prediction_logits = dec_prediction_logits = None

        loss_fct = CrossEntropyLoss(ignore_index=-100)

        # Prepare cls mask
        if self.training and random.random() < self.config.mask_cls_prob:
            # Mask some positions (from back to front)
            cls_len = cls_type_ids.sum(dim=-1)
            masked_cls_mask = torch.zeros_like(cls_type_ids, dtype=torch.float)
            for idx, length in enumerate(cls_len):
                reserved_length = random.randint(1, length)
                masked_cls_mask[idx, :reserved_length] = 1
        else:
            masked_cls_mask = cls_type_ids.float()
        masked_cls_mask = masked_cls_mask.unsqueeze(-1)

        # Encoder LM prediction
        prediction_logits = self.lm_predictions(last_hidden_state)

        if cal_mlm_loss:
            labels_ = labels.clone()
            labels_[input_ids != mask_token_id] = -100
            mlm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels_.view(-1))
            loss = mlm_loss

        # SOP task prediction
        masked_cls_len = masked_cls_mask.sum(dim=1)  # [B, 1]
        cls_hiddens = (last_hidden_state * masked_cls_mask).sum(dim=1)  # [B, D]
        cls_hiddens = cls_hiddens / masked_cls_len
        seq_relationship_logits = self.seq_relationship(cls_hiddens)

        if cal_nsp_loss:
            sop_loss = loss_fct(seq_relationship_logits.view(-1, 2), sentence_pair_label.view(-1))
            loss = sop_loss if (loss is None) else loss + sop_loss

        # Decoder LM prediction
        inputs_embeds = None

        if self.non_casual_decoder or self.casual_decoder:
            dummy_input_ids = attention_mask * mask_token_id + (1 - attention_mask) * self.config.pad_token_id
            dummy_embeds = self.bert.embeddings.forward(dummy_input_ids, token_type_ids, position_ids)  # [B, L, D]
            inputs_embeds = (
                    self.cls_proj(last_hidden_state) * masked_cls_mask + dummy_embeds * (1 - masked_cls_mask)
            )
            # Add dropout noise on cls
            inputs_embeds = nn.functional.dropout(inputs_embeds, self.config.hidden_dropout_prob, self.training)

        if self.non_casual_decoder:
            ndec_mlm_loss, ndec_prediction_logits = self.non_casual_decoder.forward(
                hidden_states=inputs_embeds,
                attention_mask=attention_mask,
                head_mask=head_mask,
                labels=labels,
            )
            loss = ndec_mlm_loss if (loss is None) else loss + ndec_mlm_loss

        if self.casual_decoder and (labels is not None):
            max_cls_len = masked_cls_len.sum(dim=-1).max().long()
            encoder_hidden_states = inputs_embeds[:, :max_cls_len].contiguous()  # [B, L', D]
            encoder_attention_mask = masked_cls_mask[:, :max_cls_len].squeeze(-1).long().contiguous()  # [B, L']
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

            dec_lm_loss, dec_prediction_logits = self.casual_decoder.forward(
                input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
            )
            loss = dec_lm_loss if (loss is None) else loss + dec_lm_loss

        if not return_dict:
            output = (prediction_logits,) + outputs[2:]
            return ((loss, mlm_loss, sop_loss, ndec_mlm_loss, dec_mlm_loss) + output) if loss is not None else output

        return LogLMForPreTrainingOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            sop_loss=sop_loss,
            ndec_mlm_loss=ndec_mlm_loss,
            dec_mlm_loss=dec_mlm_loss,
            prediction_logits=prediction_logits,
            ndec_prediction_logits=ndec_prediction_logits,
            dec_prediction_logits=dec_prediction_logits,
            seq_relationship_logits=seq_relationship_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            position_ids: torch.Tensor = None,
            cls_type_ids: torch.Tensor = None,
            # bos_token_id: Optional[int] = None,
            # pad_token_id: Optional[int] = None,
            # eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#generation
        assert self.casual_decoder is not None

        # bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        # pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # First pass: nsp_loss + mlm_loss
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        last_hidden_state = outputs.last_hidden_state

        max_cls_len = cls_type_ids.sum(dim=-1).max()
        encoder_outputs = last_hidden_state[:, :max_cls_len] * cls_type_ids[:, :max_cls_len].float().unsqueeze(dim=-1)

        # model_kwargs["encoder_outputs"] = encoder_outputs
        # return super().generate(**model_kwargs)

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_beam_sample_gen_mode = (
                (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained genertation.")

            if do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
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
