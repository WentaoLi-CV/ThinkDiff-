from transformers import (
    Blip2Config,
    T5Config,
    Blip2VisionModel,
    Blip2Processor
)
from transformers.utils import logging, ModelOutput
import torch
from torch import nn
import re
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
from thinkdiff.common.registry import registry
from thinkdiff.models.base_model import BaseModel
from torch.nn import functional as F
import random
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput, Blip2PreTrainedModel
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5LayerNorm

from thinkdiff.models.model_utils import (
    EmptyConfig,
    IdentityMap,
    )
import warnings
from torch.nn import CrossEntropyLoss
import json
import logging as py_logging
from thinkdiff.common.dist_utils import is_main_process

logger = logging.get_logger(__name__)

def _parse_entities(entities):
    if entities is None:
        return []
    if isinstance(entities, str):
        try:
            return json.loads(entities)
        except json.JSONDecodeError:
            return []
    if isinstance(entities, list):
        return entities
    return []
    
def build_vision_projector(config):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if "t5_norm" in projector_type:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu_t5_norm$', projector_type)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)

    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            if "t5_norm" in projector_type:
                layer_norm = T5LayerNorm(config.hidden_size)
            else:
                layer_norm = nn.LayerNorm(config.hidden_size)
            # with torch.no_grad():
            #     layer_norm.bias.fill_(0.0)
            #     layer_norm.weight.fill_(0.1)
            modules.append(layer_norm)
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class BlipT5DecoderConfig(Blip2Config):
    def __init__(self, max_txt_len=32, mm_projector_type="mlp2x_gelu", vision_downsample_factor=None, **kwargs):
        super().__init__(**kwargs)
        self.max_txt_len = max_txt_len
        self.mm_projector_type = mm_projector_type
        self.vision_downsample_factor = vision_downsample_factor


class T5ForDecoder(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        extra_attention_mask: Optional[torch.FloatTensor] = None,
        extra_encoder_outputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            if extra_attention_mask is not None:
                extra_attention_mask = extra_attention_mask.to(self.decoder.first_device)
            if extra_encoder_outputs_embeds is not None:
                extra_encoder_outputs_embeds = extra_encoder_outputs_embeds.to(self.decoder.first_device)

        
        if extra_attention_mask is not None:
            attention_mask = torch.cat([extra_attention_mask, attention_mask], dim=1)
        if extra_encoder_outputs_embeds is not None:
            hidden_states = torch.cat([extra_encoder_outputs_embeds, hidden_states], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def random_split_string(s):
    words = s.split(" ")  # Split the string into words
    if len(words) <= 1:
        return "", s  # If there's only one word or none, return the string as is
    split_point = random.randint(1, len(words) - 1)  # Randomly choose a split point
    part1 = ' '.join(words[:split_point])  # First part
    part2 = ' '.join(words[split_point:])  # Second part
    return part1, part2

@registry.register_model("blip-vision-t5-decoder")
class BlipVisionT5DecoderForConditionalGeneration(Blip2PreTrainedModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_blip_vision_t5_decoder": "configs/models/blip_vision_t5_decoder.yaml",
    }
    config_class = BlipT5DecoderConfig
    main_input_name = "pixel_values"

    def __init__(self, config: BlipT5DecoderConfig):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        mm_projector_config = EmptyConfig()
        
        mm_projector_config.mm_projector_type = config.mm_projector_type
        # mm_projector_config.mm_hidden_size = config.text_config.hidden_size
        mm_projector_config.mm_hidden_size = config.vision_config.hidden_size
        mm_projector_config.hidden_size = config.text_config.hidden_size

        self.mm_projector = build_vision_projector(mm_projector_config)
        
        if config.use_decoder_only_language_model:
            raise NotImplementedError("Decoder only language model is not supported yet.")
        else:
            language_model = T5ForDecoder._from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        self.tokenizer = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward_inner(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        nonzero_qformer_output_token_num: Optional[int] = None,
        nonzero_qformer_input_token_num: Optional[int] = None
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            image_embeds = vision_outputs[0]

            if self.config.vision_downsample_factor is not None:
                pooled_image_embeds = image_embeds[:, 0:1, :]
                image_embeds = image_embeds[:, 1:, :]
                # downsample image embeddings by vision_downsample_factor
                h = int(image_embeds.size(1) ** (0.5))
                w = h
                image_embeds = image_embeds.reshape(image_embeds.shape[0], h, w, image_embeds.shape[-1])
                image_embeds = image_embeds.permute(0, 3, 1, 2)
                image_embeds = F.interpolate(
                    image_embeds,
                    size=(h // self.config.vision_downsample_factor, w // self.config.vision_downsample_factor),
                    mode="bilinear",
                    align_corners=False,
                )
                image_embeds = image_embeds.permute(0, 2, 3, 1)
                image_embeds = image_embeds.reshape(image_embeds.shape[0], -1, image_embeds.shape[-1])

                image_embeds = torch.cat([pooled_image_embeds, image_embeds], dim=1)

        # step 3: use the language model, conditioned on the query outputs and the prompt
        # language_model_inputs = self.language_projection(image_embeds)
        language_model_inputs = self.mm_projector(image_embeds)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.config.use_decoder_only_language_model:
            raise NotImplementedError("Decoder only language model is not supported yet.")
        else:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                extra_attention_mask=language_model_attention_mask,
                extra_encoder_outputs_embeds=language_model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, None, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=None,
            language_model_outputs=outputs,
        )

    
    def forward(self, samples, reduction='mean'):
        pixel_values = samples["image"]
        answer = samples["answer"]
        device = pixel_values.device
        
        text_input = []
        text_output = []
        entity_stats = []
        use_entity_span_corruption = getattr(self, "use_entity_span_corruption", False)
        entity_mask_prob = getattr(self, "entity_mask_prob", 0.0)
        max_entity_spans_to_mask = getattr(self, "max_entity_spans_to_mask", 0)
        entities_list = samples.get("entities", None)
        for idx, answer_i in enumerate(answer):
            entities_i = None
            if entities_list is not None:
                entities_i = entities_list[idx]
            if use_entity_span_corruption and entities_i is not None:
                (
                    text_input_i,
                    text_output_i,
                    stat_i,
                ) = self._build_entity_span_corruption(
                    answer_i,
                    entities_i,
                    mask_prob=entity_mask_prob,
                    max_spans=max_entity_spans_to_mask,
                )
                entity_stats.append(stat_i)
            else:
                text_input_i, text_output_i = random_split_string(answer_i)
            text_input.append(text_input_i)
            text_output.append(text_output_i)

        if entity_stats and "iters" in samples and "epoch" in samples:
            if is_main_process() and samples["iters"] == 0:
                avg_entities = sum(s["entities"] for s in entity_stats) / len(entity_stats)
                avg_masked = sum(s["masked_spans"] for s in entity_stats) / len(entity_stats)
                avg_ratio = sum(s["masked_token_ratio"] for s in entity_stats) / len(entity_stats)
                skipped = sum(s["skipped_spans"] for s in entity_stats)
                py_logging.info(
                    "Entity span corruption stats - entities/sample: %.2f, masked spans/sample: %.2f, "
                    "masked token ratio: %.4f, skipped spans: %d",
                    avg_entities,
                    avg_masked,
                    avg_ratio,
                    skipped,
                )

        input_tokens = self.tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.config.max_txt_len,
            return_tensors="pt",
        ).to(device)
        output_tokens = self.tokenizer(
            text_output,
            padding="longest",
            truncation=True,
            max_length=self.config.max_txt_len,
            return_tensors="pt",
        ).to(device)

        attention_mask = input_tokens["attention_mask"]
        input_ids = input_tokens["input_ids"]
        decoder_attention_mask = output_tokens["attention_mask"]
        # decoder_input_ids = output_tokens["input_ids"]

        labels = output_tokens["input_ids"].masked_fill(
            output_tokens["input_ids"] == self.tokenizer.pad_token_id, -100
        )

        outputs = self.forward_inner(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        loss = outputs[0]

        return {"loss": loss}

    def _build_entity_span_corruption(self, caption, entities, mask_prob=0.0, max_spans=0):
        parsed_entities = sorted(
            _parse_entities(entities),
            key=lambda ent: ent.get("start", 0) if isinstance(ent, dict) else 0,
        )
        spans = []
        skipped = 0
        last_end = -1
        for ent in parsed_entities:
            if not isinstance(ent, dict):
                skipped += 1
                continue
            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None:
                skipped += 1
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                skipped += 1
                continue
            if start < 0 or end <= start or end > len(caption):
                skipped += 1
                continue
            if start < last_end:
                skipped += 1
                continue
            spans.append((start, end))
            last_end = end

        num_entities = len(parsed_entities)
        total_tokens = len(self.tokenizer(caption, add_special_tokens=False).input_ids)
        if mask_prob <= 0 or max_spans <= 0 or total_tokens == 0:
            text_input, text_output = random_split_string(caption)
            return text_input, text_output, {
                "entities": num_entities,
                "masked_spans": 0,
                "masked_token_ratio": 0.0,
                "skipped_spans": skipped,
            }

        target_tokens = max(1, int(round(total_tokens * mask_prob)))

        word_spans = []
        cursor = 0
        for word in caption.split():
            idx = caption.find(word, cursor)
            if idx == -1:
                continue
            start = idx
            end = idx + len(word)
            word_spans.append((start, end))
            cursor = end

        def _overlaps(a, b):
            return not (a[1] <= b[0] or b[1] <= a[0])

        non_entity_spans = []
        for w_span in word_spans:
            if any(_overlaps(w_span, e_span) for e_span in spans):
                continue
            non_entity_spans.append(w_span)

        random.shuffle(spans)
        random.shuffle(non_entity_spans)

        selected = []
        masked_tokens = 0

        def _span_token_len(span):
            span_text = caption[span[0]:span[1]]
            return len(self.tokenizer(span_text, add_special_tokens=False).input_ids)

        for span in spans:
            if len(selected) >= max_spans or masked_tokens >= target_tokens:
                break
            span_tokens = _span_token_len(span)
            if span_tokens == 0:
                continue
            selected.append(span)
            masked_tokens += span_tokens

        for span in non_entity_spans:
            if len(selected) >= max_spans or masked_tokens >= target_tokens:
                break
            if any(_overlaps(span, chosen) for chosen in selected):
                continue
            span_tokens = _span_token_len(span)
            if span_tokens == 0:
                continue
            selected.append(span)
            masked_tokens += span_tokens

        if not selected:
            text_input, text_output = random_split_string(caption)
            return text_input, text_output, {
                "entities": num_entities,
                "masked_spans": 0,
                "masked_token_ratio": 0.0,
                "skipped_spans": skipped,
            }

        selected = sorted(selected, key=lambda s: s[0])

        masked_input_parts = []
        label_parts = []
        cursor = 0
        for idx, (start, end) in enumerate(selected):
            masked_input_parts.append(caption[cursor:start])
            masked_input_parts.append(f"<extra_id_{idx}>")
            span_text = caption[start:end]
            label_parts.append(f"<extra_id_{idx}>")
            label_parts.append(span_text)
            cursor = end

        masked_input_parts.append(caption[cursor:])
        masked_input = "".join(masked_input_parts)
        labels = " ".join(part for part in label_parts if part)
        masked_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0.0

        return masked_input, labels, {
            "entities": num_entities,
            "masked_spans": len(selected),
            "masked_token_ratio": masked_ratio,
            "skipped_spans": skipped,
        }
        
    @classmethod
    def from_config(cls, cfg):

        dtype = cfg.get("dtype", "float32")
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        blip2_pretrained_model_name_or_path = cfg.get("blip2_pretrained_model_name_or_path", "Salesforce/blip2-flan-t5-xxl")
        
        model_config = BlipT5DecoderConfig.from_pretrained(blip2_pretrained_model_name_or_path)

        model_config.max_txt_len = cfg.get("max_txt_len", 32)
        model_config.mm_projector_type = cfg.get("mm_projector_type", "mlp2x_gelu")
        model_config.vision_downsample_factor = cfg.get("vision_downsample_factor", None)

        model = BlipVisionT5DecoderForConditionalGeneration.from_pretrained(
            blip2_pretrained_model_name_or_path,
            config=model_config,
            torch_dtype=torch.float32,
            # output_loading_info=True
        )

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model.tokenizer = processor.tokenizer
        
        if cfg.get("layer_norm_reinit_weight_with_language_encoder", False):
            with torch.no_grad():
                for module in model.mm_projector.modules():
                    if isinstance(module, T5LayerNorm):
                        # get the last layer norm weight from the language model encoder
                        module.load_state_dict(model.language_model.encoder.final_layer_norm.state_dict())
                        print("Reinit T5LayerNorm with language encoder")

        model.vision_model.to(dtype=dtype)
        model.language_model.to(dtype=dtype)

        if cfg.get("layer_norm_reinit_weight", None):
            with torch.no_grad():
                for module in model.mm_projector.modules():
                    if isinstance(module, nn.LayerNorm):
                        module.weight.fill_(cfg.layer_norm_reinit_weight)
                        module.bias.fill_(0.0)


        if cfg.get("freeze_vision", True):
            for param in model.vision_model.parameters():
                param.requires_grad = False
        
        if cfg.get("freeze_language", True):
            for param in model.language_model.parameters():
                param.requires_grad = False

        ckpt_path = cfg.get("ckpt", "")  # load weights of ClipT5
        if ckpt_path:
            print("Load BlipT5Decoder Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
    

    def forward_encoder(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        nonzero_qformer_output_token_num: Optional[int] = None,
        nonzero_qformer_input_token_num: Optional[int] = None
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            image_embeds = vision_outputs[0]

            if self.config.vision_downsample_factor is not None:
                pooled_image_embeds = image_embeds[:, 0:1, :]
                image_embeds = image_embeds[:, 1:, :]
                # downsample image embeddings by vision_downsample_factor
                h = int(image_embeds.size(1) ** (0.5))
                w = h
                image_embeds = image_embeds.reshape(image_embeds.shape[0], h, w, image_embeds.shape[-1])
                image_embeds = image_embeds.permute(0, 3, 1, 2)
                image_embeds = F.interpolate(
                    image_embeds,
                    size=(h // self.config.vision_downsample_factor, w // self.config.vision_downsample_factor),
                    mode="bilinear",
                    align_corners=False,
                )
                image_embeds = image_embeds.permute(0, 2, 3, 1)
                image_embeds = image_embeds.reshape(image_embeds.shape[0], -1, image_embeds.shape[-1])

                image_embeds = torch.cat([pooled_image_embeds, image_embeds], dim=1)

        # step 3: use the language model, conditioned on the query outputs and the prompt
        # language_model_inputs = self.language_projection(query_output)
        language_model_inputs = self.mm_projector(image_embeds)
        
        return language_model_inputs
