from __future__ import annotations

import torch

from microtrain.model import ModelConfig, SmolLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_model_shape_and_tied_embeddings() -> None:
    model = SmolLM(tiny_config())
    tokens = torch.randint(0, 64, (3, 7))
    assert model(tokens).shape == (3, 7, 64)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_causal_attention_does_not_observe_future_tokens() -> None:
    torch.manual_seed(1)
    model = SmolLM(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 20, 21]])
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])


def test_cached_and_full_forward_match() -> None:
    torch.manual_seed(2)
    model = SmolLM(tiny_config()).eval()
    tokens = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        full = model(tokens)
        cache = None
        pieces = []
        for position in range(tokens.shape[1]):
            logits, cache = model.forward_cached(tokens[:, position : position + 1], cache)
            pieces.append(logits)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)


def test_chunked_cached_forward_matches_full_forward() -> None:
    torch.manual_seed(3)
    model = SmolLM(tiny_config()).eval()
    tokens = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        full = model(tokens)
        first, cache = model.forward_cached(tokens[:, :6])
        second, _ = model.forward_cached(tokens[:, 6:], cache)
    torch.testing.assert_close(full, torch.cat((first, second), dim=1), atol=1e-5, rtol=1e-5)

