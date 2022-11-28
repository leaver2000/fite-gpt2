import os
import re
import enum
from pathlib import Path
from typing import NamedTuple, TypeAlias

import torch
from transformers import AddedToken


__all__ = [
    "SpecialTokens",
    "RegexPatterns",
    # "get_model_name",
    # Path constants and functions
    "ROOT_DATASET_PATH",
    "ROOT_MODEL_PATH",
    "MAX_LENGTH",
    "BATCH_SIZE",
    "get_file_system",
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


ModelName: TypeAlias = str
RawTextFile: TypeAlias = Path
JSONLinesFile: TypeAlias = Path
DatasetPath: TypeAlias = Path
TokenizerPath: TypeAlias = Path
ModelPath: TypeAlias = Path


class FileMap(NamedTuple):
    name: ModelName
    raw_text: RawTextFile
    jsonl: JSONLinesFile
    dataset: DatasetPath
    tokenizer: TokenizerPath
    model: ModelPath


def get_file_system(base_model: str, version: str) -> dict[str, FileMap]:
    fs = {}
    dataset_names = {
        file.stem.rstrip("-training-data") for file in ROOT_DATA_PATH.iterdir()
    }
    for name in dataset_names:
        model_name = f"{base_model}-{name}-{version}"
        fs[name] = FileMap(
            name=model_name,
            raw_text=ROOT_DATA_PATH / f"{name}-training-data.txt",
            jsonl=ROOT_DATA_PATH / f"{name}-training-data.jsonl",
            dataset=ROOT_DATASET_PATH / model_name,
            tokenizer=ROOT_TOKENIZER_PATH / model_name,
            model=ROOT_MODEL_PATH / model_name,
        )
    return fs


"""a mapping of FileMaps to the various paths used in the project"""

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


# def get_paths(model_name: str) -> tuple[ModelPath, Path, DatasetPath]:
#     paths = (
#         ROOT_MODEL_PATH / model_name,
#         ROOT_TOKENIZER_PATH / model_name,
#         ROOT_DATASET_PATH / model_name,
#     )

#     for path in paths:
#         if path.exists() and _path_is_empty(path):
#             # remove the path if it exists and is empty
#             path.rmdir()
#     return paths


# def get_model_name(base_model: str, dataset_name: str, version: str = VERSION) -> str:
#     return f"{base_model}-{dataset_name}-{version}"


class TokenEnum(str, enum.Enum):
    def __str__(self) -> str:
        return str(self.value)

    def escape(self) -> str:
        return re.escape(str(self.value))

    @property
    def compile(self) -> re.Pattern[str]:
        return re.compile(self.escape())

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
