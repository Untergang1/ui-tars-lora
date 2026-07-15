"""xFormers attention patch for Qwen2.5-VL's frozen vision encoder."""

from __future__ import annotations

import types
from typing import Optional, Tuple

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision


def _xformers_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Replace dense SDPA with a block-diagonal, autograd-capable xFormers op."""
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if position_embeddings is None:
        if rotary_pos_emb is None:
            raise ValueError("Qwen vision attention requires rotary position embeddings")
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos().float(), emb.sin().float()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    # xFormers receives [batch, sequence, heads, head_dim] and avoids an L^2 matrix.
    q, k, v = (tensor.unsqueeze(0) for tensor in (q, k, v))
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens, kv_seqlen=None)
    output = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=0.0, scale=None)
    return self.proj(output.squeeze(0).reshape(seq_length, -1))


def enable_xformers_vision_attention(model: torch.nn.Module) -> int:
    """Patch each Qwen vision block only for this Python process."""
    patched = 0
    for block in model.visual.blocks:
        block.attn.forward = types.MethodType(_xformers_forward, block.attn)
        patched += 1
    return patched
