import os
import re
from enum import Enum
from typing import Literal
from pathlib import Path

import pandas as pd
import torch
from transformers import AddedToken
from transformers import BatchEncoding

# datasets imports
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset, Batch

__all__ = [
    "get_dataset_path",
    "get_model_path",
    "get_raw_text_data",
    "get_prepared_data",
    "get_device",
    "create_tokenized_dataset",
    "PipeLineTask",
    "SpecialTokens",
    "unpack_paths",
]
STORE = Path.cwd() / "store"
DATASET_PATH = STORE / "datasets"
MODEL_PATH = STORE / "models"
RAW_TEXT = STORE / "training-data-v2.txt"
TOKEN_PATTERN = re.compile(
    r"(?<=\s\d{3})(?=\d{2,3})|(?=KT)|(?=G\d{2}KT)|(?=G\d{3}KT)|(?<=FEW|SCT|BKN|OVC)|(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB)"
)


class TafRegex:
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
    sub = TOKEN_PATTERN.sub


ModelPath = Path
DatasetPath = Path


def unpack_paths(model_name: str, version: str) -> tuple[ModelPath, DatasetPath]:
    """Unpacks the paths for the model and dataset"""
    model_path = get_model_path(model_name, version)
    dataset_path = get_dataset_path(model_name, version)
    return model_path, dataset_path


def path_is_empty(path: Path) -> bool:
    return path.exists() and not os.listdir(path)


def get_dataset_path(dataset_prep_method: str, version: str) -> Path:
    path = DATASET_PATH / f"{dataset_prep_method}-v{version}"
    # if  path.exists() and path_is_empty(path) :
    #     path.rmdir()
    return path


def get_model_path(model_name: str, version: str) -> Path:
    path = MODEL_PATH / f"{model_name}-v{version}"
    if path_is_empty(path):
        path.rmdir()
    return path


def get_raw_text_data() -> str:
    with RAW_TEXT.open("r") as f:
        return f.read().strip()


DSPrepMethod = Literal["taf-full", "taf-line", "taf-component"]


def get_prepared_data(method: DSPrepMethod) -> pd.DataFrame:
    text_data = get_raw_text_data()

    if method == "taf-full":
        # each TAF is split as a new line
        text_data = text_data.split("\n\n###\n\n")
    elif method == "taf-line":
        # each line of every TAF is split as a new line
        text_data = text_data.replace("###", "").splitlines()
    elif method == "taf-component":
        # splitting individual text components for tokenization
        # ie: 12015G20KT 9999 BKN030CB -> 120 15 G 20 KT 9999 BKN 030 CB
        text_data = TOKEN_PATTERN.sub(" ", text_data).splitlines()
    else:
        raise ValueError("method must be either 'element' or 'line'")

    text_data = [line.strip() for line in text_data if line]
    labels = [0 if line.startswith("TAF") else 1 for line in text_data]
    df = pd.DataFrame({"text": text_data, "labels": labels})

    return df.loc[df.text != "", :]


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
    # drop the training data from the original dataframe to create the test data
    test = df.drop(train.index)
    # return the train and test dataframes
    return train, test


def create_tokenized_dataset(prep_method: DSPrepMethod, batch_size: int) -> None:
    # load data from the raw text file
    df = get_prepared_data(prep_method)

    # split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2)
    # create a DatasetDict
    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )
    # batch encoder for tokenization
    def batch_encode(batch: Batch) -> BatchEncoding:
        return tokenizer(batch["text"], truncation=True)  # type: ignore

    # tokenize dataset with batch encoding using the tokenizer
    ds = ds.map(
        batch_encode,
        batched=True,
        batch_size=batch_size,
    )
    ds.save_to_disk(DATASET_PATH)  # type: ignore



AddedTokenType = str | AddedToken


class TokenEnum(str, Enum):
    value: AddedToken

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def to_dict(cls) -> dict[str, AddedTokenType]:
        return {k: v.value for k, v in cls._member_map_.items()}

    @classmethod
    def include(cls, *tokens: AddedToken) -> dict[str, AddedTokenType]:
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
    bos_token = AddedToken("<|bos|>")
    """A special token representing the beginning of a sentence. Will be associated to `self.bos_token` and added to `self.special_tokens_map`."""
    pad_token = AddedToken("<|pad|>")
    eos_token = AddedToken("<|eos|>")
    sep_token = AddedToken("<|sep|>")
    """A special token representing the end of a sentence. Will be associated to `self.eos_token` and"""
