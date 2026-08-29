from __future__ import annotations


class CharTokenizer:
    """A reversible ASCII tokenizer for network-free unit tests."""

    bos_id = 0
    eos_id = 0
    pad_id = 0
    vocab_size = 129

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [ord(character) + 1 for character in text]
        return ([0] if bos else []) + ids + ([0] if eos else [])

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(chr(token - 1) for token in ids if token != 0)

