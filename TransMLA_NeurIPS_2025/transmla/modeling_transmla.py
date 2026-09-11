"""Portable, inference-only MLA for the Qwen3 and MiMo conversion paths.

This file is copied into exported Hugging Face checkpoints. Qwen3 retains its
nonlinear per-head Q/K normalization before rotation/compression; it is not a
stock DeepSeek checkpoint. Both paths cache only latent KV and decoupled RoPE.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel, Qwen3RMSNorm,
)


class TransMLAConfig(Qwen3Config):
    model_type = "transmla"

    def __init__(self, kv_lora_rank=512, qk_mqa_dim=64, q_lora_rank=None,
                 qk_norm_preserved=True, qk_latent_layernorm=False,
                 source_model_type="qwen3", o_proj_bias=False,
                 mla_query_chunk_size=64, mla_score_budget_mb=128,
                 mla_prefill_backend="auto", **kwargs):
        super().__init__(**kwargs)
        self.kv_lora_rank = kv_lora_rank
        self.qk_mqa_dim = qk_mqa_dim
        self.q_lora_rank = q_lora_rank
        self.qk_norm_preserved = qk_norm_preserved
        self.qk_latent_layernorm = qk_latent_layernorm
        self.source_model_type = source_model_type
        self.o_proj_bias = o_proj_bias
        self.mla_query_chunk_size = mla_query_chunk_size
        self.mla_score_budget_mb = mla_score_budget_mb
        self.mla_prefill_backend = mla_prefill_backend
        if mla_query_chunk_size <= 0 or mla_score_budget_mb <= 0:
            raise ValueError("MLA attention chunk size and score budget must be positive")
        if mla_prefill_backend not in ("auto", "chunked"):
            raise ValueError("mla_prefill_backend must be auto or chunked")
        if qk_mqa_dim <= 0 or self.head_dim % qk_mqa_dim or qk_mqa_dim % 2:
            raise ValueError("qk_mqa_dim must be even and divide head_dim")
        self.collapse = self.head_dim // qk_mqa_dim
        width = self.num_key_value_heads * self.head_dim
        if not 0 < kv_lora_rank <= 2 * width - qk_mqa_dim:
            raise ValueError("kv_lora_rank exceeds the joint non-RoPE K/V dimension")
        if qk_norm_preserved and (q_lora_rank is not None or qk_latent_layernorm):
            raise ValueError("Qwen3 preserves source Q/K norms; query LoRA and latent norms are unsupported")
        if "sliding_attention" in self.layer_types:
            raise ValueError("This conversion backend supports full-attention source layers only")


def apply_mla_rope(x, cos, sin):
    """TransMLA's interleaved projection order to the HF split-half RoPE order."""
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2).transpose(-1, -2).reshape(shape)
    first, second = x.chunk(2, dim=-1)
    return x * cos.unsqueeze(1) + torch.cat((-second, first), -1) * sin.unsqueeze(1)


class TransMLAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.latent_dim = config.num_key_value_heads * self.head_dim
        self.rope_dim = config.qk_mqa_dim
        self.rank = config.kv_lora_rank
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.qk_norm_preserved = config.qk_norm_preserved
        h, d, r = self.num_heads, self.head_dim, self.rope_dim
        bias = config.attention_bias
        if self.qk_norm_preserved:
            self.q_proj = nn.Linear(config.hidden_size, h * d, bias=bias)
            self.k_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=bias)
            self.v_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=bias)
            self.q_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
            self.k_rotation = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            self.kv_down_proj = nn.Linear(2 * self.latent_dim - r, self.rank, bias=False)
            self.register_buffer("q_rope_weight", torch.zeros(h, d, r))
            self.register_buffer("balance_scale", torch.ones(()))
            self.scaling = d ** -0.5
        else:
            if config.q_lora_rank is None:
                self.q_proj = nn.Linear(config.hidden_size, h * (d + r), bias=bias)
            else:
                self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
                self.q_b_proj = nn.Linear(config.q_lora_rank, h * (d + r), bias=bias)
                if config.qk_latent_layernorm:
                    self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.rank + r, bias=bias)
            if config.qk_latent_layernorm:
                self.kv_a_layernorm = nn.RMSNorm(self.rank, eps=config.rms_norm_eps)
            # The upstream conversion already folds the compensating scale into Q.
            self.scaling = (d + r) ** -0.5
        self.kv_b_proj = nn.Linear(self.rank, h * (2 * d), bias=False)
        self.o_proj = nn.Linear(h * d, config.hidden_size, bias=config.o_proj_bias)

    def project(self, hidden_states):
        b, t, _ = hidden_states.shape
        if self.qk_norm_preserved:
            query = self.q_norm(self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim))
            key = self.k_norm(self.k_proj(hidden_states).view(b, t, -1, self.head_dim)).flatten(-2)
            key = self.k_rotation(key)
            value = self.v_proj(hidden_states)
            k_rope = key[..., :self.rope_dim]
            joint = torch.cat((key[..., self.rope_dim:] / self.balance_scale, value), -1)
            latent = self.kv_down_proj(joint)
            q_rope = torch.einsum("bthd,hdr->bhtr", query, self.q_rope_weight)
            q_nope = query.transpose(1, 2)
        else:
            if self.config.q_lora_rank is None:
                query = self.q_proj(hidden_states)
            else:
                query = self.q_a_proj(hidden_states)
                if hasattr(self, "q_a_layernorm"):
                    query = self.q_a_layernorm(query)
                query = self.q_b_proj(query)
            query = query.view(b, t, self.num_heads, -1).transpose(1, 2)
            q_nope, q_rope = query.split((self.head_dim, self.rope_dim), -1)
            latent, k_rope = self.kv_a_proj_with_mqa(hidden_states).split((self.rank, self.rope_dim), -1)
            if hasattr(self, "kv_a_layernorm"):
                latent = self.kv_a_layernorm(latent)
        return q_nope, q_rope, latent.unsqueeze(1), k_rope.unsqueeze(1)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, past_key_value=None,
                output_attentions=False, **kwargs):
        if self.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Portable TransMLA supports eager/SDPA attention only")
        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is not None and not isinstance(cache, DynamicCache):
            raise ValueError("Portable TransMLA requires DynamicCache, which stores latent-sized tensors")
        q_nope, q_rope, latent, k_rope = self.project(hidden_states)
        cos, sin = (x[..., ::self.config.collapse] for x in position_embeddings)
        q_rope = apply_mla_rope(q_rope, cos, sin)
        k_rope = apply_mla_rope(k_rope, cos, sin)
        if cache is not None:
            # DynamicCache's two slots store latent KV and RoPE K, respectively.
            latent, k_rope = cache.update(latent, k_rope, self.layer_idx)
        key_up, value_up = self.kv_b_proj.weight.view(self.num_heads, 2 * self.head_dim, self.rank).split(self.head_dim, 1)
        # Prefill may reconstruct temporary per-head K/V for FlashAttention;
        # only the compressed latent and RoPE key above are retained in cache.
        output = self._flash_prefill(q_nope, q_rope, latent, k_rope, key_up, value_up,
                                     attention_mask, output_attentions)
        if output is not None:
            return self.o_proj(output.transpose(1, 2).flatten(-2)), None
        q_latent = torch.einsum("bhtd,hdr->bhtr", q_nope, key_up)
        query = torch.cat((q_latent, q_rope), -1)
        key = torch.cat((latent, k_rope), -1)
        q_len, k_len = query.shape[-2], key.shape[-2]
        positions = cache_position if cache_position is not None else torch.arange(k_len - q_len, k_len, device=query.device)
        # A score tile (in FP32) is capped independently of sequence length.
        # Broadcast the single latent head directly; do not repeat K/V 32 times.
        budget = int(self.config.mla_score_budget_mb * 1024**2)
        chunk = max(1, min(self.config.mla_query_chunk_size,
                           budget // (4 * query.shape[0] * self.num_heads * k_len)))
        if output_attentions and 4 * query.shape[0] * self.num_heads * q_len * k_len > budget:
            raise ValueError("output_attentions exceeds the diagnostic score budget")
        outputs, all_weights = [], []
        key_positions = torch.arange(k_len, device=query.device)
        score_key = key.transpose(-1, -2).float()
        for start in range(0, q_len, chunk):
            stop = min(q_len, start + chunk)
            allowed = key_positions[None, :] <= positions[start:stop, None]
            # Casting a BF16 matmul result is too late: large common score
            # offsets can already have erased the differences softmax needs.
            with torch.autocast(device_type=query.device.type, enabled=False):
                scores = (query[..., start:stop, :].float() @ score_key) * self.scaling
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    allowed = allowed[None, None] & attention_mask[:, None, None, :k_len].bool()
                elif attention_mask.ndim == 4:
                    mask = attention_mask[..., :k_len]
                    if mask.shape[-2] != 1:
                        mask = mask[..., start:stop, :]
                    if mask.dtype == torch.bool:
                        allowed = allowed & mask
                    else:
                        scores = scores + mask.float()
                        allowed = allowed & (mask > torch.finfo(mask.dtype).min)
                else:
                    raise ValueError("attention_mask must be a 2D padding or 4D attention mask")
            scores = scores.masked_fill(~allowed, float("-inf"))
            weights = scores.softmax(-1).nan_to_num().to(query.dtype)
            weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
            outputs.append(weights @ latent)
            if output_attentions:
                all_weights.append(weights)
        self.last_attention_backend = "chunked_latent"
        self.last_score_tile_shape = (query.shape[0], self.num_heads, min(chunk, q_len), k_len)
        output = torch.cat(outputs, dim=-2)
        weights = torch.cat(all_weights, dim=-2) if output_attentions else None
        output = torch.einsum("bhtr,hdr->bthd", output, value_up)
        output = self.o_proj(output.flatten(-2))
        return output, weights

    def _expanded_qkv(self, q_nope, q_rope, latent, k_rope, key_up, value_up):
        query = torch.cat((q_nope, q_rope), -1)
        key = torch.einsum("bstr,hdr->bhtd", latent, key_up)
        key = torch.cat((key, k_rope.expand(-1, self.num_heads, -1, -1)), -1)
        value = torch.einsum("bstr,hdr->bhtd", latent, value_up)
        # Torch Flash SDPA requires equal Q/K/V dimensions. The padded output
        # components are zero and are discarded; preserve the original scale.
        return query, key, F.pad(value, (0, self.rope_dim))

    def _flash_prefill(self, q_nope, q_rope, latent, k_rope, key_up, value_up,
                       attention_mask, output_attentions):
        if (self.config.mla_prefill_backend != "auto" or output_attentions
                or self.config._attn_implementation == "eager"
                or q_nope.device.type != "cuda" or q_nope.dtype not in (torch.float16, torch.bfloat16)
                or q_nope.shape[-2] <= 1 or q_nope.shape[-2] != latent.shape[-2]
                or self.head_dim + self.rope_dim > 256):
            return None
        if attention_mask is not None and (attention_mask.ndim != 2 or not bool(attention_mask.all())):
            return None
        from torch.backends.cuda import SDPAParams, can_use_flash_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel
        query, key, value = self._expanded_qkv(q_nope, q_rope, latent, k_rope, key_up, value_up)
        dropout = self.attention_dropout if self.training else 0.0
        if not can_use_flash_attention(SDPAParams(query, key, value, None, dropout, True, False)):
            return None
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                    scale=self.scaling, dropout_p=dropout)
        self.last_attention_backend = "flash_expanded_prefill"
        self.last_score_tile_shape = None
        return output[..., :self.head_dim]


class TransMLAModel(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        for index, layer in enumerate(self.layers):
            layer.self_attn = TransMLAAttention(config, index)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Keep the original 2D padding mask. HF's default mask builder can
        # allocate a full Q x K tensor before attention has a chance to tile it.
        if not isinstance(attention_mask, dict):
            attention_mask = {"full_attention": attention_mask}
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class TransMLAForCausalLM(Qwen3ForCausalLM):
    config_class = TransMLAConfig
    _supports_flash_attn = False
    _supports_flex_attn = False

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = TransMLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


TransMLAConfig.register_for_auto_class()
TransMLAForCausalLM.register_for_auto_class("AutoModelForCausalLM")
