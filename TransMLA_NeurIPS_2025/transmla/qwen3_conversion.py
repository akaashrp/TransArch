"""Qwen3 adaptation: perform TransMLA's PCA after the source Q/K norms.

The nonlinear normalization is retained at inference. Unlike the linear source
path, these weights cannot be represented by stock DeepSeek's projections.
"""

import torch
from torch import nn
from .partial_rope import PartialRope
from .modeling_transmla import TransMLAAttention
from .utils import pca_calc


class QKNormPartialRope(PartialRope):
    def __init__(self, source, key_outputs, freqfold=1, collapse=1, rope_head=1):
        super().__init__(source, freqfold=freqfold, collapse=collapse, rope_head=rope_head)
        self.q_norm = source.q_norm
        self.k_norm = source.k_norm
        self.k_input_proj = source.k_proj
        # Rotate normalized activations, rather than incorrectly rotating the
        # source linear weight on the other side of a nonlinear normalization.
        self.k_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False,
                                device=source.k_proj.weight.device, dtype=source.k_proj.weight.dtype)
        with torch.no_grad():
            self.k_proj.weight.copy_(torch.eye(self.latent_dim, device=self.k_proj.weight.device, dtype=self.k_proj.weight.dtype))
        rotation = self.joint_complex_pca(key_outputs, freqfold)
        self.rotate_k_proj(rotation, freqfold)
        self.rotate_k_up_proj(rotation, freqfold)

    def project_qkv(self, hidden_states):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(shape)).flatten(-2)
        key = self.k_norm(self.k_input_proj(hidden_states).view(shape)).flatten(-2)
        return query, self.k_proj(key), self.v_proj(hidden_states)


@torch.no_grad()
def compress_qwen3_attention(partial, key_outputs, value_outputs, config, balance_kv_ratio=1.0):
    d, r, rank = partial.latent_dim, config.qk_mqa_dim, config.kv_lora_rank
    dtype, device = partial.q_proj.weight.dtype, partial.q_proj.weight.device
    if balance_kv_ratio is None:
        alpha = torch.tensor(1.0)
    else:
        key_norm = torch.cat([x.reshape(-1, d)[:, r:] for x in key_outputs]).float().norm(dim=0).mean()
        value_norm = torch.cat([x.reshape(-1, d) for x in value_outputs]).float().norm(dim=0).mean()
        if not torch.isfinite(key_norm + value_norm) or min(key_norm, value_norm) <= 0:
            raise ValueError("KV balancing requires finite, nonzero calibration activations")
        alpha = key_norm / (value_norm * balance_kv_ratio)
    joint = [torch.cat((key[..., r:] / alpha, value), -1) for key, value in zip(key_outputs, value_outputs)]
    basis = pca_calc(joint, device)[:, :rank]
    key_up = partial.k_up_proj.weight[:, r:].double() * alpha.to(device)
    value_up = partial.v_up_proj.weight.double()
    key_up = key_up @ basis[:d-r]
    value_up = value_up @ basis[d-r:]
    attention = TransMLAAttention(config, partial.layer_idx).to(device=device, dtype=dtype)
    attention.q_proj = partial.q_proj
    attention.k_proj = partial.k_input_proj
    attention.v_proj = partial.v_proj
    attention.o_proj = partial.o_proj
    attention.q_norm = partial.q_norm
    attention.k_norm = partial.k_norm
    attention.k_rotation = partial.k_proj
    attention.balance_scale.copy_(alpha)
    attention.q_rope_weight.copy_(partial.k_up_proj.weight[:, :r].reshape(config.num_attention_heads, config.head_dim, r))
    attention.kv_down_proj.weight.copy_(basis.T)
    up = torch.cat((key_up.view(config.num_attention_heads, config.head_dim, rank),
                    value_up.view(config.num_attention_heads, config.head_dim, rank)), dim=1)
    attention.kv_b_proj.weight.copy_(up.flatten(0, 1))
    return attention.eval()
