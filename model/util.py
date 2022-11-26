import os
import re

from enum import Enum
from pathlib import Path
from typing import TypeVar, ParamSpec

from transformers import AddedToken
from typing import Literal

import torch
from transformers import GPT2LMHeadModel, GPT2Config

from . import __version__ as VERSION

from .typing import StrPath, ModelPath, DatasetPath

P = ParamSpec("P")
R = TypeVar("R")

__all__ = [
    "SpecialTokens",
    "RegexPatterns",
    "get_model_name",
    # Path constants and functions
    "get_paths",
    "ROOT_DATASET_PATH",
    "ROOT_MODEL_PATH",
    "JSONL_FILE_MAP",
    "MAX_LENGTH",
    "BATCH_SIZE",
]
# RUNTIME VARIABLES
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 256))
STORE = Path.cwd() / os.getenv("STORE_PATH", "store")
ROOT_MODEL_PATH = STORE / "models"
ROOT_DATASET_PATH = STORE / "datasets"
ROOT_TOKENIZER_PATH = STORE / "tokenizer"
ROOT_DATA_PATH = STORE / "data"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
JSONL_FILE_MAP = {
    "taf": ROOT_DATA_PATH / "taf-training-data.jsonl",
}
BaseModelNames = Literal["gpt2"]
DatasetNames = Literal["taf"]


TOKEN_PATTERN = re.compile(
    r"(?<=\s\d{3})(?=\d{2,3})|(?=KT)|(?=G\d{2}KT)|(?=G\d{3}KT)|(?<=FEW|SCT|BKN|OVC)|(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB)"
)


class RegexPatterns:
    # split-winds: 23015G25KT -> 230 15 G 25 KT
    WIND_GUST = r"""
    (?<=\s(\d{3}|VRB))(?=\d{2,3}(KT|G\d{2,3}KT)) # wind direction and speed
    |(?=G\d{2,3}KT\s) # before Gust
    |(?<=G)(?=\d{2,3}KT\s) # after Gust
    |(?=KT\s) # before KT
    """
    # split-clouds: SCT250 -> SCT 250MODEL_PATH
    CLOUD_COVER = r"""
    (?<=FEW|SCT|BKN|OVC)(?=\d{3}) # after cloud cover
    |(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB) # before CB
    """
    TOKEN_PATTERN = re.compile(r"|".join([WIND_GUST, CLOUD_COVER]), re.VERBOSE)
    TEMPERATURE_GROUP = r"\sTX(M)?\d{2}\/\d{4}Z\sTN(M)?\d{2}\/\d{4}Z"
    sub = TOKEN_PATTERN.sub


def _path_is_empty(path: Path) -> bool:
    return path.exists() and not os.listdir(path)


def get_paths(model_name: str) -> tuple[ModelPath, Path, DatasetPath]:
    paths = (
        ROOT_MODEL_PATH / model_name,
        ROOT_TOKENIZER_PATH / model_name,
        ROOT_DATASET_PATH / model_name,
    )

    for path in paths:
        if path.exists() and _path_is_empty(path):
            # remove the path if it exists and is empty
            path.rmdir()
    return paths


def get_model_name(
    base_model: BaseModelNames, dataset_name: DatasetNames, version: str = VERSION
) -> str:
    return f"{base_model}-{dataset_name}-{version}"


def get_language_model(
    base_model: StrPath,
    config: GPT2Config | None = None,
    verbose: bool = False,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu", index=0
    ),
) -> GPT2LMHeadModel:

    model = GPT2LMHeadModel.from_pretrained(
        base_model,
        config=config,
    )
    # should always be in eval mode
    assert isinstance(model, GPT2LMHeadModel)
    if verbose:
        print(model.config)
    return model.to(device)


class TokenEnum(str, Enum):
    value: AddedToken

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def to_dict(cls) -> dict[str, AddedToken]:
        return {k: v.value for k, v in cls._member_map_.items()}

    @classmethod
    def include(cls, *tokens: AddedToken) -> dict[str, AddedToken]:
        special_tokens = cls.to_dict().copy()
        if tokens:
            special_tokens["additional_special_tokens"] = list(tokens)  # type: ignore
        return special_tokens

    def escape(self) -> str:
        return re.escape(str(self))

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.escape())


class SpecialTokens(TokenEnum):
    """Special tokens that can be added to a tokenizer"""

    cls_token = "<|cls|>"
    """
    A special token representing the beginning of a sentence.
    Will be associated to `self.bos_token` and added to `self.special_tokens_map`."""
    pad_token = "<|pad|>"
    """
    A special token representing a padding token.
    Will be associated to `self.pad_token` and added to `self.special_tokens_map`."""
    bos_token = "<|bos|>"
    """
    A special token representing the beginning of a sentence.
    Will be associated to `self.bos_token` and added to `self.special_tokens_map`."""
    eos_token = "<|eos|>"
    """
    A special token representing the end of a sentence.
    Will be associated to `self.eos_token` and added to `self.special_tokens_map`."""
    sep_token = "<|sep|>"
    """
    A special token representing the end of a sentence.
    Will be associated to `self.eos_token` and added to `self.special_tokens_map`."""
    unk_token = "<|unk|>"


def dedent_plus(text: str) -> str:
    return "\n".join(t.strip() for t in text.split("\n"))
