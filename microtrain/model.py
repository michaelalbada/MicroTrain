"""The SmolLM2 transformer, written directly in PyTorch.

This file intentionally implements one architecture rather than a configurable
model zoo. Parameter names mirror Hugging Face's Llama implementation so the
public SmolLM2-135M safetensors checkpoint can be loaded without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LayerCache = tuple[Tensor, Tensor]
KVCache = list[LayerCache]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 49_152
    hidden_size: int = 576
    intermediate_size: int = 1_536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 8_192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100_000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        names = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in names})

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(dtype)


def rotary_frequencies(config: ModelConfig, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return Llama-style rotary cos/sin tensors shaped [batch, time, head_dim]."""

    inverse = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=position_ids.device, dtype=torch.float32) / config.head_dim)
    )
    angles = position_ids.float().unsqueeze(-1) * inverse
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [batch, heads, time, head_dim]; cos/sin: [batch, time, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: Tensor,
        past: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerCache | None]:
        batch, time, _ = x.shape
        head_dim = self.config.head_dim

        q = self.q_proj(x).view(batch, time, self.config.num_attention_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.config.num_key_value_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.config.num_key_value_heads, head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past is not None:
            past_k, past_v = past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v) if use_cache else None
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        k_for_attention = k.repeat_interleave(repeats, dim=1)
        v_for_attention = v.repeat_interleave(repeats, dim=1)

        key_positions = torch.arange(k.shape[2], device=x.device)
        causal_mask = key_positions.view(1, 1, 1, -1) <= position_ids.view(batch, 1, time, 1)
        attended = F.scaled_dot_product_attention(
            q,
            k_for_attention,
            v_for_attention,
            attn_mask=causal_mask,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.o_proj(attended), present


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: Tensor,
        past: LayerCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, LayerCache | None]:
        attention, present = self.self_attn(
            self.input_layernorm(x), cos, sin, position_ids, past, use_cache
        )
        x = x + attention
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class SmolLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        cache: KVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, KVCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, time = input_ids.shape
        past_length = 0 if cache is None else cache[0][0].shape[2]
        if cache is not None and len(cache) != len(self.model.layers):
            raise ValueError("cache must have one entry per transformer layer")
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + time, device=input_ids.device
            ).expand(batch, -1)
        elif position_ids.shape != (batch, time):
            raise ValueError("position_ids must match input_ids shape")
        if int(position_ids.max()) >= self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")

        x = self.model.embed_tokens(input_ids)
        cos, sin = rotary_frequencies(self.config, position_ids, x.dtype)
        new_cache: KVCache | None = [] if use_cache else None
        for index, layer in enumerate(self.model.layers):
            past = None if cache is None else cache[index]
            x, present = layer(x, cos, sin, position_ids, past, use_cache)
            if new_cache is not None and present is not None:
                new_cache.append(present)
        return self.lm_head(self.model.norm(x)), new_cache

    def forward(self, input_ids: Tensor, position_ids: Tensor | None = None) -> Tensor:
        logits, _ = self._forward(input_ids, position_ids, cache=None, use_cache=False)
        return logits

    def forward_cached(
        self,
        input_ids: Tensor,
        cache: KVCache | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, KVCache]:
        logits, new_cache = self._forward(input_ids, position_ids, cache, use_cache=True)
        assert new_cache is not None
        return logits, new_cache

