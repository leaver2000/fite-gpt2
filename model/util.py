import os
import re
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import torch

#

from transformers import AddedToken

__all__ = [
    "SpecialTokens",
    "RegexPatterns",
    "unpack_paths",
    # Path constants
    "DATASET_PATH",
    "MODEL_PATH",
]
STORE = Path.cwd() / "store"
DATASET_PATH = STORE / "datasets"
MODEL_PATH = STORE / "models"
# RAW_TEXT_FILE = STORE / "training-data-v2.txt"

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
    # split-clouds: SCT250 -> SCT 250
    CLOUD_COVER = r"""
    (?<=FEW|SCT|BKN|OVC)(?=\d{3}) # after cloud cover
    |(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB) # before CB
    """
    TOKEN_PATTERN = re.compile(r"|".join([WIND_GUST, CLOUD_COVER]), re.VERBOSE)
    TEMPERATURE_GROUP = r"\sTX(M)?\d{2}\/\d{4}Z\sTN(M)?\d{2}\/\d{4}Z"
    sub = TOKEN_PATTERN.sub


ModelPath: TypeAlias = Path
DatasetPath: TypeAlias = Path
JSONLinesFile: TypeAlias = Path
TEXTFile: TypeAlias = Path


def path_is_empty(path: Path) -> bool:
    return path.exists() and not os.listdir(path)


def unpack_paths(
    dataset_name: str,
    version: str,
) -> tuple[ModelPath, DatasetPath, TEXTFile, JSONLinesFile]:
    """Unpacks the paths for the model and dataset"""
    text_file = STORE / f"{dataset_name}-training-data.txt"
    json_lines = STORE / f"{dataset_name}-training-data.jsonl"
    # only the model receives the version suffix
    paths = (MODEL_PATH / f"{dataset_name}-{version}", DATASET_PATH / dataset_name)

    for path in paths:
        if path.exists() and path_is_empty(path):
            # remove the path if it exists and is empty
            path.rmdir()
    return *paths, text_file, json_lines


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # split the training data as a fraction of test_size
    train = df.sample(frac=1 - test_size, random_state=0)
    # drop the training data from the original DataFrames to create the test data
    test = df.drop(train.index)
    # return the train and test DataFrames
    return train, test


AddedTokenType = str | AddedToken


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


class SpecialTokens(TokenEnum):
    """Special tokens that can be added to a tokenizer"""

    cls_token = AddedToken("<|TAF|>")
    """
    A special token representing the beginning of a sentence.
    Will be associated to `self.bos_token` and added to `self.special_tokens_map`."""
    pad_token = AddedToken("<|pad|>")
    """
    A special token representing a padding token.
    Will be associated to `self.pad_token` and added to `self.special_tokens_map`."""
    bos_token = AddedToken("<|bos|>")
    """
    A special token representing the beginning of a sentence.
    Will be associated to `self.bos_token` and added to `self.special_tokens_map`."""
    eos_token = AddedToken("<|eos|>")
    """
    A special token representing the end of a sentence.
    Will be associated to `self.eos_token` and added to `self.special_tokens_map`."""
    sep_token = AddedToken("<|sep|>")
    """
    A special token representing the end of a sentence.
    Will be associated to `self.eos_token` and added to `self.special_tokens_map`."""
    unk_token = AddedToken("<|unk|>")


def dedent_plus(text: str) -> str:
    return "\n".join(t.strip() for t in text.split("\n"))
