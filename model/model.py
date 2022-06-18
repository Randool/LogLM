"""
@File        :  model.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/16
@Version     :  1.0
@Description :  Bert with variable length of `[CLS]`
"""
from typing import Optional, Tuple

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertLMPredictionHead


@dataclass
class LogLMForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    seq_relationship_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LogLMPreTrainedModel(BertPreTrainedModel):
    pass


class LogLMForPreTraining(LogLMPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)

        self.prompt_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh())
        self.lm_predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_predictions.decoder = new_embeddings

    def _forward(
            self,
            inputs_embeds,
            prompt_type_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        cal_mlm_loss = (labels is not None)
        cal_nsp_loss = (next_sentence_label is not None)

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        prediction_logits = None
        seq_relationship_logits = None
        loss_fct = CrossEntropyLoss() if cal_mlm_loss or cal_nsp_loss else None

        if cal_mlm_loss:
            prediction_logits = self.lm_predictions(outputs.last_hidden_state)
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = masked_lm_loss

        if cal_nsp_loss:
            prompt_hiddens = (outputs.last_hidden_state * prompt_type_ids.float().unsqueeze(-1)).mean(dim=1)
            seq_relationship_logits = self.seq_relationship(prompt_hiddens)
            next_sentence_loss = loss_fct(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))
            loss = next_sentence_loss if (loss is None) else loss + next_sentence_loss

        return loss, prediction_logits, seq_relationship_logits, outputs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            prompt_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mask_token_id=103,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings.forward(input_ids, token_type_ids, position_ids)

        if (labels is not None) and (prompt_type_ids is None):
            prompt_type_ids = (1 <= input_ids) & (input_ids <= 11)
            prompt_type_ids[:, 0] = True

        # First pass: nsp_loss + mlm_loss
        loss, prediction_logits, seq_relationship_logits, outputs = self._forward(
            inputs_embeds=inputs_embeds,
            prompt_type_ids=prompt_type_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels,
            next_sentence_label=next_sentence_label,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Second pass: mlm_loss
        if labels is not None:
            real_token_mask = prompt_type_ids ^ attention_mask.bool()  # F, F, T, T, ..., T[, F, F]
            dummy_input_ids = torch.zeros_like(input_ids)
            dummy_input_ids[torch.where(real_token_mask)] = mask_token_id

            dummy_embeds = self.bert.embeddings.forward(dummy_input_ids, token_type_ids, position_ids)  # [B, L, D]

            inputs_embeds = (
                    self.prompt_proj(outputs.last_hidden_state) * prompt_type_ids.float().unsqueeze(-1)
                    + dummy_embeds * (1 - prompt_type_ids).float().unsqueeze(-1)
            )

            loss2, _prediction_logits, _seq_relationship_logits, _outputs = self._forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                prompt_type_ids=prompt_type_ids,
                head_mask=head_mask,
                labels=labels,
                next_sentence_label=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            loss = loss + loss2

        if not return_dict:
            output = (prediction_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LogLMForPreTrainingOutput(
            loss=loss,
            prediction_logits=prediction_logits,
            seq_relationship_logits=seq_relationship_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LogLMForSequenceClassification(LogLMPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout if (config.classifier_dropout is not None)
                       else config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            prompt_type_ids=None,
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

        pooled_output = (outputs.last_hidden_state * prompt_type_ids.float().unsqueeze(-1)).mean(dim=1)
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
