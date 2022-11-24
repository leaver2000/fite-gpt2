import os
import re
from enum import Enum
from pathlib import Path
from typing import ParamSpec

import pandas as pd
import torch

#

from transformers import AddedToken

__all__ = [
    "get_device",
    "PipeLineTask",
    "SpecialTokens",
    "RegexPatterns",
    "unpack_paths",
    # Path constants
    "DATASET_PATH",
    "MODEL_PATH",
    "RAW_TEXT_FILE",
]
STORE = Path.cwd() / "store"
DATASET_PATH = STORE / "datasets"
MODEL_PATH = STORE / "models"
RAW_TEXT_FILE = STORE / "training-data-v2.txt"

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


### Path Functions
P = ParamSpec("P")
ModelPath = Path
DatasetPath = Path
JSONLinesFile = Path


def path_is_empty(path: Path) -> bool:
    return path.exists() and not os.listdir(path)


def unpack_paths(version: str) -> tuple[ModelPath, DatasetPath, JSONLinesFile]:
    """Unpacks the paths for the model and dataset"""
    paths = (MODEL_PATH / version, DATASET_PATH / version)
    for path in paths:
        if path.exists() and path_is_empty(path):
            # remove the path if it exists and is empty
            path.rmdir()
    json_lines = STORE / f"training-data-{version}.jsonl"
    return *paths, json_lines


def get_device(verbose: bool = True, index=0) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=index)
    if verbose:
        device_properties = torch.cuda.get_device_properties(device=device)
        print(
            f"""
        ***** Device Properties *****
        {device_properties.name=}
        {device_properties.total_memory=:,} bytes
        {device_properties.multi_processor_count=} processors
        """
        )

    return device


class StrEnum(str, Enum):
    value: str
    _member_map_: dict[str, str]

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def _generate_next_value_(name: str, *_: tuple[int, int, tuple[int, ...]]) -> str:
        return name

    @classmethod
    def _to_dict(cls) -> dict[str, str]:
        return cls._member_map_


class PipeLineTask(StrEnum):

    TEXT_CLASSIFICATION = "text-classification"
    TEXT2TEXT_GENERATION = "text2text-generation"
    FEATURE_EXTRACTION = "feature-extraction"
    FILL_MASK = "fill-mask"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    QUESTION_ANSWERING = "question-answering"
    TABLE_QUESTION_ANSWERING = "table-question-answering"
    TEXT_GENERATION = "text-generation"
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    EXTRACTIVE_QUESTION_ANSWERING = "extractive-question-answering"


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
        """
        Represents a token that can be be added to a ~tokenizers.Tokenizer. It can have special options that defines the way it should behave.

        Args:
            content (str): The content of the token

            single_word (bool, defaults to False):
                Defines whether this token should only match single words. If True, this token will never match inside of a word. For example the token ing would match on tokenizing if this option is False, but not if it is True. The notion of "inside of a word" is defined by the word boundaries pattern in regular expressions (ie. the token should start and end with word boundaries).

            lstrip (bool, defaults to False):
                Defines whether this token should strip all potential whitespaces on its left side. If True, this token will greedily match any whitespace on its left. For example if we try to match the token [MASK] with lstrip=True, in the text "I saw a [MASK]", we would match on " [MASK]". (Note the space on the left).

            rstrip (bool, defaults to False):
                Defines whether this token should strip all potential whitespaces on its right side. If True, this token will greedily match any whitespace on its right. It works just like lstrip but on the right.

            normalized (bool, defaults to True with ~tokenizers.Tokenizer.add_tokens and False with ~tokenizers.Tokenizer.add_special_tokens):
                Defines whether this token should match against the normalized version of the input text. For example, with the added token "yesterday", and a normalizer in charge of lowercasing the text, the token could be extract from the input "I saw a lion Yesterday".
        """
        special_tokens = cls.to_dict().copy()
        if tokens:
            special_tokens["additional_special_tokens"] = list(tokens)  # type: ignore
        return special_tokens


class SpecialTokens(TokenEnum):
    """Special tokens that can be added to a tokenizer

    cls_token: The classification token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when added to sequences.
    sep_token: The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
    pad_token: The token used for padding, for example when batching sequences of different lengths.
    mask_token: The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
    unk_token: The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
    bos_token: The beginning of sentence token.
    eos_token: The end of sentence token.
    additional_special_tokens: Additional special tokens used by the tokenizer.
    """

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
