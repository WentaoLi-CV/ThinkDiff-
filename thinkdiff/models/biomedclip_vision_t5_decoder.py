import math

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
from peft import LoraConfig, get_peft_model

from thinkdiff.models.model_utils import (
    EmptyConfig,
    IdentityMap,
    )
import warnings
from torch.nn import CrossEntropyLoss


import os
import json
import timm
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoTokenizer

from thinkdiff.models.moe_context import MoEContext
from thinkdiff.models.inject_modalmoe import inject_modalmoe_linear
import gc
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from thinkdiff.models.modalmoe_linear import ModalMoELinear

logger = logging.get_logger(__name__)


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


# TODO
def _resolve_dtype(x):
    if isinstance(x, torch.dtype):
        return x
    x = str(x).lower()
    if x in ["fp16", "float16", "16", "half"]:
        return torch.float16
    if x in ["bf16", "bfloat16"]:
        return torch.bfloat16
    return torch.float32


def _get_local_rank_device():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return local_rank, device


def build_bnb_quant_config(cfg, dtype: torch.dtype):
    """只返回对 language_model 生效的 bnb config（我们不在这里量化 vision/mm_projector）"""
    quant_mode = str(cfg.get("quantization", "none")).lower()
    if quant_mode in ["none", "null", "false", "0", "no"]:
        return None

    if quant_mode in ["int8", "8bit"]:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if quant_mode in ["4bit", "nf4"]:
        compute_dtype = dtype
        if compute_dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")).lower(),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        )

    raise ValueError(f"Unknown quantization mode: {quant_mode}")


def freeze_base_keep_adapters(module: nn.Module):
    """
    冻结 base，但保留 LoRA / ModalMoE 等 adapter 可训练
    """
    for name, p in module.named_parameters():
        if ("lora_" in name) or ("moe" in name) or ("router" in name) or ("gate" in name):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def cast_trainable_to_fp32(module: nn.Module):
    for n, p in module.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)


def apply_lora_ffn_only(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[list] = None,
):
    """
    给 module 注入 LoRA（只打到 FFN 的 Linear 上）
    target_modules 用“名字子串匹配”，例如 ["fc1","fc2"] 或 ["wi","wo","wi_0","wi_1"]
    """
    if target_modules is None:
        target_modules = ["fc1", "fc2", "wi", "wo", "wi_0", "wi_1"]

    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type=None,     # 让它支持普通 nn.Module 注入（不强制是特定任务）
    )
    return get_peft_model(module, peft_cfg)


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


# TODO
def _token_spans(text: str):
    # whitespace tokenization with char offsets
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _geom_span_len(mean_span_len: int, max_len: int = 10):
    # geometric with p = 1/mean, expected length ~= mean_span_len
    if mean_span_len <= 1:
        return 1
    p = 1.0 / float(mean_span_len)
    L = 1
    while L < max_len and random.random() > p:
        L += 1
    return L


def entity_span_corrupt_t5(
    text: str,
    entities,  # list[dict] or None
    entity_mask_prob: float = 0.8,
    span_mask_ratio: float = 0.15,
    max_sentinels: int = 20,
):
    """
    T5-style span corruption:
      input : replace each masked span with <extra_id_i>
      target: <extra_id_0> span0 <extra_id_1> span1 ... <extra_id_n>
    Entity spans are preferentially masked; extra random spans fill up to span_mask_ratio.
    """
    if text is None:
        text = ""
    # text = text.strip()
    if len(text) == 0:
        return text, "<extra_id_0> <extra_id_1>"

    tok_spans = _token_spans(text)
    n_tok = len(tok_spans)
    if n_tok == 0:
        return text, "<extra_id_0> <extra_id_1>"

    # ---- parse entities (json string already handled outside if needed) ----
    ent_list = entities if isinstance(entities, list) else []
    # Map entity char spans -> token indices
    ent_token_sets = []
    for ent in ent_list:
        if not isinstance(ent, dict):
            continue
        s = int(ent.get("start", -1))
        e = int(ent.get("end", -1))
        if s < 0 or e <= s or s >= len(text):
            continue
        e = min(e, len(text))
        covered = []
        for ti, (ts, te) in enumerate(tok_spans):
            if te <= s:
                continue
            if ts >= e:
                break
            # overlap
            if ts < e and te > s:
                covered.append(ti)
        if covered:
            ent_token_sets.append(set(covered))

    masked = set()

    # ---- entity-first masking ----
    if ent_token_sets:
        random.shuffle(ent_token_sets)
        for sset in ent_token_sets:
            if len(masked) >= n_tok:
                break
            if random.random() < entity_mask_prob:
                masked |= sset
        # ensure at least one entity span masked (if any exist)
        if len(masked) == 0:
            masked |= random.choice(ent_token_sets)

    # ---- fill with random spans to reach span_mask_ratio ----
    target_mask = int(round(span_mask_ratio * n_tok))
    target_mask = max(1, min(target_mask, n_tok))

    # 由 target_mask 推导“想要多少段 span”
    K = int(round(math.sqrt(target_mask)))
    K = max(1, min(K, max_sentinels))

    # 由 K 推导 mean_span_len（不再是超参）
    mean_span_len = int(math.ceil(target_mask / K))
    mean_span_len = max(1, mean_span_len)
    if target_mask < len(masked):
        target_mask = len(masked)

    candidates = [i for i in range(n_tok) if i not in masked]
    # keep sampling spans until enough tokens masked
    tries = 0
    while len(masked) < min(target_mask, n_tok) and candidates and tries < 5 * n_tok:
        tries += 1
        start = random.choice(candidates)
        L = _geom_span_len(mean_span_len)
        end = min(n_tok, start + L)
        # ensure no overlap
        if any(i in masked for i in range(start, end)):
            candidates = [i for i in candidates if i not in masked]
            continue
        for i in range(start, end):
            masked.add(i)
        candidates = [i for i in candidates if i not in masked]

    masked_sorted = sorted(masked)
    if not masked_sorted:
        # fallback: behave like random split if nothing masked
        return random_split_string(text)

    # ---- convert masked token indices -> char spans (merge contiguous) ----
    spans = []
    st = masked_sorted[0]
    prev = st
    for idx in masked_sorted[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((tok_spans[st][0], tok_spans[prev][1]))
            st = prev = idx
    spans.append((tok_spans[st][0], tok_spans[prev][1]))

    # limit number of sentinels
    spans = spans[:max_sentinels]

    # ---- build input/target ----
    in_parts = []
    out_parts = []
    cur = 0
    for i, (s, e) in enumerate(spans):
        sentinel = f"<extra_id_{i}>"
        in_parts.append(text[cur:s])
        in_parts.append(sentinel)
        out_parts.append(sentinel)
        out_parts.append(text[s:e])
        cur = e
    in_parts.append(text[cur:])
    # final sentinel (T5 convention)
    out_parts.append(f"<extra_id_{len(spans)}>")

    def _clean(x):
        x = re.sub(r"\s+", " ", x)
        return x.strip()

    return _clean(" ".join(in_parts)), _clean(" ".join(out_parts))


class BiomedCLIPVisionModel(nn.Module):
    """ TODO
    Offline BiomedCLIP vision tower:
    - reads <biomedclip_dir>/open_clip_config.json + open_clip_pytorch_model.bin
    - builds timm ViT trunk (vit_base_patch16_224)
    - returns token sequence like HF vision models:
        BaseModelOutput(last_hidden_state = [B, 1+Np, C])
    """
    def __init__(self, biomedclip_dir: str, align_to_16x16: bool = False):
        super().__init__()
        self.biomedclip_dir = biomedclip_dir
        self.align_to_16x16 = align_to_16x16

        cfg_path = os.path.join(biomedclip_dir, "open_clip_config.json")
        ckpt_path = os.path.join(biomedclip_dir, "open_clip_pytorch_model.bin")
        assert os.path.isfile(cfg_path), f"Missing: {cfg_path}"
        assert os.path.isfile(ckpt_path), f"Missing: {ckpt_path}"

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        vision_cfg = cfg["model_cfg"]["vision_cfg"]
        self.image_size = int(vision_cfg.get("image_size", 224))
        self.timm_model_name = vision_cfg["timm_model_name"]  # e.g. vit_base_patch16_224

        # Build timm trunk (NO internet)
        self.trunk = timm.create_model(
            self.timm_model_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # extract only vision trunk weights from open_clip ckpt
        vision_state = {}
        if any(k.startswith("visual.trunk.") for k in state.keys()):
            prefix = "visual.trunk."
            for k, v in state.items():
                if k.startswith(prefix):
                    vision_state[k[len(prefix):]] = v
        elif any(k.startswith("visual.") for k in state.keys()):
            prefix = "visual."
            for k, v in state.items():
                if k.startswith(prefix):
                    vision_state[k[len(prefix):]] = v
        else:
            vision_state = state  # fallback

        msg = self.trunk.load_state_dict(vision_state, strict=False)
        # 你可以调试时打印 msg.missing_keys / msg.unexpected_keys

        self.hidden_size = getattr(self.trunk, "num_features", 768)

    # @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ):
        x = self.trunk.forward_features(pixel_values)

        # timm ViT 有时返回 dict（包含 cls / patch tokens）
        if isinstance(x, dict):
            if "x_norm_patchtokens" in x:
                patch = x["x_norm_patchtokens"]                 # [B, Np, C]
                if "x_norm_clstoken" in x:
                    cls = x["x_norm_clstoken"].unsqueeze(1)     # [B, 1, C]
                else:
                    cls = patch.mean(dim=1, keepdim=True)
                x = torch.cat([cls, patch], dim=1)              # [B, 1+Np, C]
            else:
                x = next(v for v in x.values() if torch.is_tensor(v))

        # 有的实现只给全局 embedding [B,C]，补成 token 形式
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,C]

        # ===== 可选：把 patch tokens 对齐到 16x16（256 patches）以匹配你原 BLIP2 的 257 token 结构 =====
        if self.align_to_16x16 and x.size(1) > 1:
            cls, patch = x[:, :1, :], x[:, 1:, :]     # cls: [B,1,C], patch: [B,Np,C]
            n = patch.size(1)
            h = int(n ** 0.5)
            if h * h == n and h != 16:
                patch = patch.view(patch.size(0), h, h, patch.size(-1)).permute(0, 3, 1, 2)  # [B,C,h,w]
                patch = F.interpolate(patch, size=(16, 16), mode="bilinear", align_corners=False)
                patch = patch.permute(0, 2, 3, 1).reshape(patch.size(0), 16 * 16, patch.size(1))  # [B,256,C]
                x = torch.cat([cls, patch], dim=1)  # [B,257,C]

        return BaseModelOutput(last_hidden_state=x)


@registry.register_model("biomedclip-vision-t5-decoder-quantity")
class BiomedclipVisionT5DecoderForConditionalGenerationQuantity(Blip2PreTrainedModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_blip_vision_t5_decoder": "configs/models/blip_vision_t5_decoder.yaml",
    }
    config_class = BlipT5DecoderConfig
    main_input_name = "pixel_values"

    def __init__(self, config: BlipT5DecoderConfig):
        super().__init__(config)
        self.vision_model = Blip2VisionModel(config.vision_config)

        # TODO
        self.moe_ctx = None
        self.moe_aux_weight = 0.0

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

    def _vision_has_trainable(self):
        return any(p.requires_grad for p in self.vision_model.parameters())


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
        # TODO
        vision_ctx = torch.enable_grad() if self._vision_has_trainable() else torch.no_grad()
        # with torch.no_grad():
        with vision_ctx:
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
        # TODO
        if getattr(self, "moe_ctx", None) is not None:
            mid = samples.get("modality_id", None)  # LongTensor[B]
            if mid is None:
                # 容错：全当 Other
                mid = torch.full((samples["image"].size(0),), 4, device=samples["image"].device, dtype=torch.long)
            self.moe_ctx.set_modality_ids(mid, device=samples["image"].device)

        pixel_values = samples["image"]
        answer = samples["answer"]

        device = pixel_values.device
        
        text_input = []
        text_output = []
        # TODO
        # for answer_i in answer:
        #     text_input_i, text_output_i = random_split_string(answer_i)
        entities_batch = samples.get("entities", None)
        for i, answer_i in enumerate(answer):
            ents_i = []
            if entities_batch is not None:
                # entities_batch is List[str] because we json.dumps in dataset
                raw = entities_batch[i]
                if isinstance(raw, str) and len(raw) > 0:
                    try:
                        ents_i = json.loads(raw)
                    except Exception:
                        ents_i = []
                elif isinstance(raw, list):
                    ents_i = raw
            if getattr(self.config, "mask_strategy", "random_split") == "entity_span_t5":
                text_input_i, text_output_i = entity_span_corrupt_t5(
                    answer_i,
                    ents_i,
                    entity_mask_prob=getattr(self.config, "entity_mask_prob", 0.8),
                    span_mask_ratio=getattr(self.config, "span_mask_ratio", 0.15),
                    max_sentinels=getattr(self.config, "max_sentinels", 20),
                )
            else:
                text_input_i, text_output_i = random_split_string(answer_i)

            text_input.append(text_input_i)
            text_output.append(text_output_i)

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
        # TODO
        # loss = outputs[0]
        #
        # # TODO
        # if getattr(self, "moe_ctx", None) is not None and self.moe_aux_weight != 0.0:
        #     moe_aux = self.moe_ctx.pop_aux_loss()  # float32 scalar
        #     loss = loss + moe_aux.to(loss.dtype) * float(self.moe_aux_weight)
        #
        # return {"loss": loss}

        t5_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss = t5_loss
        moe_aux = None
        if getattr(self, "moe_ctx", None) is not None and self.moe_aux_weight != 0.0:
            moe_aux = self.moe_ctx.pop_aux_loss()  # float32 scalar, with graph
            loss = loss + moe_aux.to(loss.dtype) * float(self.moe_aux_weight)
        return {
            "loss": loss,
            "t5_loss": t5_loss,
            "aux_loss": moe_aux if moe_aux is not None else torch.zeros((), device=loss.device, dtype=torch.float32),
        }

    @classmethod
    def from_config(cls, cfg):

        # ===== 0) dtype & device =====
        dtype = _resolve_dtype(cfg.get("dtype", "float32"))
        local_rank, device = _get_local_rank_device()

        # ===== 1) quant config =====
        quant_cfg = build_bnb_quant_config(cfg, dtype)

        # ===== 2) load config =====
        t5_src = cfg.get("t5_model_name_or_path", None)
        blip2_pretrained_model_name_or_path = cfg.get("blip2_pretrained_model_name_or_path", "Salesforce/blip2-flan-t5-xxl")
        model_config = BlipT5DecoderConfig.from_pretrained(blip2_pretrained_model_name_or_path)
        model_config.max_txt_len = cfg.get("max_txt_len", 32)
        model_config.mm_projector_type = cfg.get("mm_projector_type", "mlp2x_gelu")
        model_config.vision_downsample_factor = cfg.get("vision_downsample_factor", None)

        # ===== 3) load base model (do NOT quantize vision by default) =====

        model = BiomedclipVisionT5DecoderForConditionalGenerationQuantity.from_pretrained(
            blip2_pretrained_model_name_or_path,
            config=model_config,
            torch_dtype=dtype,  # 用 bf16/fp16 降低加载显存压力
        )

        # ===== 4) tokenizer =====
        if t5_src:
            model.tokenizer = AutoTokenizer.from_pretrained(t5_src, use_fast=False)
        else:
            # 没量化时也可以继续用 BLIP2 processor 的 tokenizer
            processor = Blip2Processor.from_pretrained(blip2_pretrained_model_name_or_path)
            model.tokenizer = processor.tokenizer

        # ===== 5) mask configs =====
        model.config.mask_strategy = cfg.get("mask_strategy", "random_split")
        model.config.entity_mask_prob = float(cfg.get("entity_mask_prob", 0.8))
        model.config.span_mask_ratio = float(cfg.get("span_mask_ratio", 0.15))
        model.config.max_sentinels = int(cfg.get("max_sentinels", 20))
        model.noise_density = float(cfg.get("noise_density", 0.15))
        model.entity_ratio = float(cfg.get("entity_ratio", 0.7))
        model.max_mask_entities = int(cfg.get("max_mask_entities", 3))

        # ===== 6) optional swap vision: BiomedCLIP =====
        biomedclip_dir = cfg.get("biomedclip_model_dir", "")
        if biomedclip_dir:
            align_16x16 = cfg.get("biomedclip_align_to_16x16", False)
            print(f"[BiomedCLIP] Using local vision encoder: {biomedclip_dir}, align_16x16={align_16x16}")
            model.vision_model = BiomedCLIPVisionModel(biomedclip_dir, align_to_16x16=align_16x16)

        # ===== 9) freeze everything first =====
        for p in model.parameters():
            p.requires_grad_(False)

        # ===== 10) rebuild mm_projector as fp32 trainable =====
        old_proj_sd = None
        try:
            old_proj_sd = model.mm_projector.state_dict()
        except Exception:
            old_proj_sd = None

        mm_cfg = EmptyConfig()
        mm_cfg.mm_projector_type = model_config.mm_projector_type
        vision_hidden = getattr(model.vision_model, "hidden_size", None)
        if vision_hidden is None:
            vision_hidden = model_config.vision_config.hidden_size
        t5_hidden = int(
            getattr(model.language_model.config, "d_model",
                    getattr(model.language_model.config, "hidden_size", model_config.text_config.hidden_size)))
        mm_cfg.mm_hidden_size = int(vision_hidden)
        mm_cfg.hidden_size = int(t5_hidden)
        model.mm_projector = build_vision_projector(mm_cfg).to(device=device, dtype=torch.float32)
        for p in model.mm_projector.parameters():
            p.requires_grad_(True)

        if bool(cfg.get("projector_keep_pretrained", True)) and old_proj_sd is not None:
            try:
                model.mm_projector.load_state_dict(old_proj_sd, strict=False)
                print("[Projector] loaded old weights (strict=False)")
            except Exception as e:
                print(f"[Projector] skip loading old weights: {e}")

        if cfg.get("layer_norm_reinit_weight_with_language_encoder", False):
            with torch.no_grad():
                for module in model.mm_projector.modules():
                    if isinstance(module, T5LayerNorm):
                        module.load_state_dict(model.language_model.encoder.final_layer_norm.state_dict())
                        print("Reinit T5LayerNorm with language encoder")

        # ===== 11) vision ModalMoE (trainable) =====
        use_lora_vision = cfg.get("use_lora_vision", False)
        vision_impl = cfg.get("vision_lora_impl", "peft")  # "peft" or "modalmoe" or "none"
        if use_lora_vision and vision_impl == "modalmoe":
            num_modalities = int(cfg.get("num_modalities", 5))  # CT/X-ray/MRI/US/Other
            model.moe_ctx = MoEContext(num_modalities=num_modalities)
            model.moe_aux_weight = float(cfg.get("moe_aux_weight", 1.0))

            vision_backbone = model.vision_model.trunk if hasattr(model.vision_model, "trunk") else model.vision_model
            replaced = inject_modalmoe_linear(
                vision_backbone,
                ctx=model.moe_ctx,
                target_modules=cfg.get("modalmoe_target_modules", ["fc1", "fc2"]),
                r=int(cfg.get("modalmoe_r", 8)),
                lora_alpha=int(cfg.get("modalmoe_alpha", 16)),
                lora_dropout=float(cfg.get("modalmoe_dropout", 0.05)),
                group_mode=True,
                num_modalities=num_modalities,
                shared_experts=int(cfg.get("shared_experts", 2)),
                modality_experts=int(cfg.get("modality_experts", 2)),
                gate_embed_dim=int(cfg.get("gate_embed_dim", 0)),
                gate_hidden_dim=int(cfg.get("gate_hidden_dim", 128)),
                wspec_low=float(cfg.get("wspec_low", 0.6)),
                wspec_high=float(cfg.get("wspec_high", 0.8)),
                wspec_prior_weight=float(cfg.get("wspec_prior_weight", 0.0)),
                lbc_weight=float(cfg.get("lbc_weight", 0.0)),
                enable_g5=bool(cfg.get("enable_g5", False)),
                g5_weight=float(cfg.get("g5_weight", 0.0)),
            )
            print(f"[ModalMoE] replaced {len(replaced)} linears, e.g. {replaced[:5]}")

            if hasattr(model.vision_model, "trunk"):
                model.vision_model.trunk = vision_backbone
            else:
                model.vision_model = vision_backbone

        elif use_lora_vision and vision_impl == "peft":
            vision_backbone = model.vision_model.trunk if hasattr(model.vision_model, "trunk") else model.vision_model
            vision_backbone = apply_lora_ffn_only(
                vision_backbone,
                r=int(cfg.get("lora_vision_r", 8)),
                alpha=int(cfg.get("lora_vision_alpha", 16)),
                dropout=float(cfg.get("lora_vision_dropout", 0.0)),
                target_modules=cfg.get("lora_vision_target_modules", ["fc1", "fc2"]),
            )
            if hasattr(model.vision_model, "trunk"):
                model.vision_model.trunk = vision_backbone
            else:
                model.vision_model = vision_backbone

        # ===== build quantized language_model (T5) =====
        if quant_cfg is not None:
            try:
                del model.language_model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.language_model = T5ForDecoder.from_pretrained(
                t5_src,
                quantization_config=quant_cfg,
                device_map={"": local_rank},
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

            # k-bit 训练准备（会处理 layernorm、requires_grad、cast 等）
            model.language_model = prepare_model_for_kbit_training(
                model.language_model,
                use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
            )

            # 更新 t5_hidden（因为 language_model 已替换）
            t5_hidden = int(getattr(model.language_model.config, "d_model", model_config.text_config.hidden_size))

        # ===== 12) text encoder LoRA (trainable, PEFT switch) =====
        use_lora_text = bool(cfg.get("use_lora_text", False))
        if use_lora_text:
            if quant_cfg is not None:
                lcfg = LoraConfig(
                    r=int(cfg.get("lora_text_r", 8)),
                    lora_alpha=int(cfg.get("lora_text_alpha", 16)),
                    lora_dropout=float(cfg.get("lora_text_dropout", 0.0)),
                    bias="none",
                    target_modules=cfg.get("lora_text_target_modules", ["wi", "wo", "wi_0", "wi_1"]),
                    task_type="SEQ_2_SEQ_LM",
                )
                model.language_model = get_peft_model(model.language_model, lcfg)

        if not bool(cfg.get("lora_text_apply_to_decoder", False)):
            for n, p in model.language_model.named_parameters():
                if ("decoder" in n) and ("lora_" in n):
                    p.requires_grad_(False)

        # ===== 13) move non-quant modules to device/dtype =====
        if torch.cuda.is_available():
            model.mm_projector.to(device=device, dtype=torch.float32)
            model.vision_model.to(device=device, dtype=dtype)

        # ===== 14) freeze base keep adapters =====
        if cfg.get("freeze_vision", True):
            freeze_base_keep_adapters(model.vision_model)
        if cfg.get("freeze_language", True):
            freeze_base_keep_adapters(model.language_model)
        for p in model.mm_projector.parameters():
            p.requires_grad_(True)

        # ===== 15) trainable dtype control =====
        cast_trainable_to_fp32(model)
        if cfg.get("layer_norm_reinit_weight", None):
            with torch.no_grad():
                for module in model.mm_projector.modules():
                    if isinstance(module, nn.LayerNorm):
                        module.weight.fill_(cfg.layer_norm_reinit_weight)
                        module.bias.fill_(0.0)

        # ===== 16) load ckpt =====
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load BlipT5Decoder Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)

        # ===== debug trainables =====
        train_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print("Trainable params:", len(train_names))
        for n in train_names[:50]:
            print("  ", n)

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
