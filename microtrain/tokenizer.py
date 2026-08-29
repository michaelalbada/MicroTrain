"""Thin wrapper around the tokenizer released with SmolLM2."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class MicroTokenizer:
    def __init__(self, tokenizer: object, bos_id: int = 0, eos_id: int = 0) -> None:
        self._tokenizer = tokenizer
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = eos_id

    @classmethod
    def from_file(cls, path: str | Path, bos_id: int = 0, eos_id: int = 0) -> "MicroTokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'tokenizers' package to load SmolLM2's tokenizer") from exc
        return cls(Tokenizer.from_file(str(path)), bos_id=bos_id, eos_id=eos_id)

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = list(self._tokenizer.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)  # type: ignore[attr-defined]

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())  # type: ignore[attr-defined]

