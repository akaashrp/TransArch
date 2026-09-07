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
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RMSNorm


class TransMLAConfig(Qwen3Config):
    model_type = "transmla"

    def __init__(self, kv_lora_rank=512, qk_mqa_dim=64, q_lora_rank=None,
                 qk_norm_preserved=True, qk_latent_layernorm=False,
                 source_model_type="qwen3", o_proj_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.kv_lora_rank = kv_lora_rank
        self.qk_mqa_dim = qk_mqa_dim
        self.q_lora_rank = q_lora_rank
        self.qk_norm_preserved = qk_norm_preserved
        self.qk_latent_layernorm = qk_latent_layernorm
        self.source_model_type = source_model_type
        self.o_proj_bias = o_proj_bias
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
        q_latent = torch.einsum("bhtd,hdr->bhtr", q_nope, key_up)
        query = torch.cat((q_latent, q_rope), -1)
        key = torch.cat((latent, k_rope), -1)
        q_len, k_len = query.shape[-2], key.shape[-2]
        if attention_mask is None:
            positions = cache_position if cache_position is not None else torch.arange(k_len - q_len, k_len, device=query.device)
            attention_mask = torch.arange(k_len, device=query.device)[None, :] <= positions[:, None]
        else:
            attention_mask = attention_mask[..., :k_len]
        weights = None
        if self.config._attn_implementation == "eager" or output_attentions:
            scores = (query @ key.transpose(-1, -2)) * self.scaling
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask
            weights = scores.float().softmax(-1).nan_to_num().to(query.dtype)
            weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
            output = weights @ latent
        else:
            output = F.scaled_dot_product_attention(
                query, key, latent, attn_mask=attention_mask,
                scale=self.scaling, enable_gqa=True,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
        output = torch.einsum("bhtr,hdr->bthd", output, value_up)
        output = self.o_proj(output.flatten(-2))
        return output, weights


class TransMLAForCausalLM(Qwen3ForCausalLM):
    config_class = TransMLAConfig
    _supports_flash_attn = False
    _supports_flex_attn = False

    def __init__(self, config):
        super().__init__(config)
        for index, layer in enumerate(self.model.layers):
            layer.self_attn = TransMLAAttention(config, index)
        self.post_init()


TransMLAConfig.register_for_auto_class()
TransMLAForCausalLM.register_for_auto_class("AutoModelForCausalLM")
