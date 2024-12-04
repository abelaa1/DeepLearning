# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:21:08 2024

@author: Kirill
"""

import math
import deepspeed
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, KLDivLoss

import transformers
from transformers.models.opt import OPTModel, OPTForSequenceClassification, OPTForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)

class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim, lora_alpha, dropout, training):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)
        self.non_linearity = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)
        self.dropout = dropout
        self.training = training

        if lora_alpha == -1:
            self.scale = torch.nn.Parameter(torch.ones(1))
        else:
            self.scale = lora_alpha

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.down(x)
        x = self.non_linearity(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.up(x)
        x = torch.mul(x, self.scale)
        return x

class MLPAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim, dropout, training):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)
        self.non_linearity = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)
        self.dropout = dropout
        self.training = training
        self.scale = torch.nn.Parameter(torch.ones(1))

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.down(x)
        x = self.non_linearity(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.up(x)
        x = torch.mul(x, self.scale)
        return x

class ScalingAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return torch.mul(x, self.scale)

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.word_embed_proj_dim, config.word_embed_proj_dim, bias=True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(config.word_embed_proj_dim, config.num_labels, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        self.dense.weight.data.normal_(mean=0.0, std=config.init_std)
        self.dense.bias.data.zero_()
        self.output.weight.data.normal_(mean=0.0, std=config.init_std)
        self.output.bias.data.zero_()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class OPTAttention(transformers.models.opt.modeling_opt.OPTAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_decoder: bool = False, bias: bool = True, adapter_type: str = None, adapter_dim: int = None, lora_alpha: float = None):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.adapter_type = adapter_type
        self.adapter_dim = adapter_dim
        self.lora_alpha = lora_alpha

        if self.adapter_type == "lora":
            self.query_adapter = LoRAAdapter(embed_dim, adapter_dim, lora_alpha, dropout=self.dropout, training=self.training)
            self.value_adapter = LoRAAdapter(embed_dim, adapter_dim, lora_alpha, dropout=self.dropout, training=self.training)
        elif self.adapter_type == "ia3":
            self.key_adapter = ScalingAdapter(embed_dim)
            self.value_adapter = ScalingAdapter(embed_dim)

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, layer_head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if self.adapter_type == "lora":
            query_states = query_states + self.query_adapter(hidden_states)

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.adapter_type == "lora":
                value_states = value_states + self.value_adapter(hidden_states)
            elif self.adapter_type == "ia3":
                key_states = key_states + self.key_adapter(hidden_states)
                value_states = value_states + self.value_adapter(hidden_states)

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}")
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class OPTWithClassifier(transformers.models.opt.modeling_opt.OPTPreTrainedModel):
    def __init__(self, config, adapter_type: str = None, adapter_dim: int = None, lora_alpha: int = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = transformers.models.opt.modeling_opt.OPTModel(config)
        self.classification_head = ClassificationHead(config)

        self.adapter_type = adapter_type
        self.adapter_dim = adapter_dim
        self.lora_alpha = lora_alpha

        for i, layer in enumerate(self.model.decoder.layers):
            layer.self_attn = OPTAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                bias=True,
                adapter_type=self.adapter_type,
                adapter_dim=self.adapter_dim,
                lora_alpha=self.lora_alpha
            )

        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.output)
        self.model._init_weights(self.classification_head.dense)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        hidden_states[eos_mask, :] = hidden_states[eos_mask, :] / eos_mask.sum()
        hidden_states = hidden_states[eos_mask, :].view(hidden_states.size(0), hidden_states.size(2))

        logits = self.classification_head(hidden_states)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the student model with adapters
student_model = OPTWithClassifier.from_pretrained(
    "facebook/opt-125m",
    adapter_type="lora",
    adapter_dim=32,
    lora_alpha=8
)

# Initialize the teacher model without adapters (assuming it's already fine-tuned)
teacher_model = OPTForSequenceClassification.from_pretrained("path/to/teacher/model")

# Define the knowledge distillation trainer
class KnowledgeDistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for student model
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits

        # Forward pass for teacher model
        with torch.no_grad():
            outputs_teacher = teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        # Calculate the distillation loss using KL divergence
        loss_fct = KLDivLoss(reduction="batchmean")
        loss = loss_fct(
            torch.nn.functional.log_softmax(logits_student / temperature, dim=-1),
            torch.nn.functional.softmax(logits_teacher / temperature, dim=-1)
        )

        return (loss, outputs_student) if return_outputs else loss

# Initialize the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the trainer
trainer = KnowledgeDistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Train the student model with knowledge distillation
trainer.train()
