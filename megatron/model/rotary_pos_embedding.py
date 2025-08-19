# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import torch

from torch import einsum, nn
from deepspeed.accelerator import get_accelerator

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

# sin, cos tensors cached for all devices
cos_cached = None
sin_cached = None

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq.to(get_accelerator().current_device_name())
        self.theta = theta
        # self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, position_ids, offset=0):
        # position_ids shape: [B, seq_length]
        if isinstance(position_ids, int):
            position_ids = torch.arange(position_ids, device=get_accelerator().current_device_name()).unsqueeze(0)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids_offset = position_ids + offset
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids_offset.transpose(0, 1)[:, None, :].float()
        angle = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((angle, angle), dim=-1).unsqueeze(2)
        # rope = (emb.cos(), emb.sin())
        return emb


# NOTE: change einops to improve efficiency
def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    t_pass = None
    if t.shape[-1] != rot_dim:
        # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    freqs_ = freqs[:t.shape[0]]
    cos = freqs_.cos().to(t.dtype)
    sin = freqs_.sin().to(t.dtype)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos) + (_rotate_half(t) * sin)
    if t_pass is None:
        return t
    return torch.cat((t, t_pass), dim=-1)
