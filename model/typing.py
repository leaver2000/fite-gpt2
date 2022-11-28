from os import PathLike
from pathlib import Path
from typing import TypeAlias
from transformers import GPT2TokenizerFast, GPT2Tokenizer

__all__ = ["StrPath", "ModelPath", "DatasetPath", "GPT2TokenizerType", "PathLike"]


ModelPath: TypeAlias = Path
DatasetPath: TypeAlias = Path
JSONLinesFile: TypeAlias = Path
TEXTFile: TypeAlias = Path

StrPath: TypeAlias = str | Path

GPT2TokenizerType: TypeAlias = GPT2TokenizerFast | GPT2Tokenizer
