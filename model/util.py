import os
import re
import enum
import dataclasses
from pathlib import Path

from typing import Iterable, Generic, Union
from typing_extensions import TypeVarTuple, Unpack

import toml
import torch
from transformers import AddedToken

from api.pipeline import HyperParameterStrategy, HyperParameters

from .typing import (
    PyProjectTOML,
    Model,
    ModelName,
    Version,
    RawTextFile,
    JSONLinesFile,
    DatasetDictPath,
    TokenizerPath,
    ModelPath,
)


Ts = TypeVarTuple("Ts")


__all__ = [
    "FileSystem",
    "FileSystemDirectory",
    "ResultRecord",
    "FITEConfig",
    "SpecialTokens",
    "ActionStr",
    "CONSTANTS",
]
# RUNTIME VARIABLES
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)


class CONSTANTS(enum.IntEnum):
    BATCH_SIZE = os.getenv("BATCH_SIZE", 8)
    MAX_LENGTH = os.getenv("MAX_LENGTH", 256)


# ModelName: TypeAlias = str
# Version: TypeAlias = str
# RawTextFile: TypeAlias = Path
# JSONLinesFile: TypeAlias = Path
# DatasetDictPath: TypeAlias = Path
# TokenizerPath: TypeAlias = Path
# ModelPath: TypeAlias = Path


@dataclasses.dataclass
class DataClassBase(Generic[Unpack[Ts]]):
    def to_dict(self) -> dict[str, Union[Unpack[Ts]]]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FITEConfig(DataClassBase[list[Model]]):
    models: list[Model]

    @classmethod
    def load(cls, path: Path):
        config = PyProjectTOML.load(path)
        models = config["project"]["models"]
        return cls(models=models)

    def get_model(self, key) -> Model:
        for model in self.models:
            if model["name"] == key:
                return model
        raise ValueError(f"Model {key} not found in config")


@dataclasses.dataclass
class FileSystem(
    DataClassBase[
        str,
        Model,
        Version,
        ModelName,
        RawTextFile,
        JSONLinesFile,
        DatasetDictPath,
        TokenizerPath,
        ModelPath,
    ]
):
    base_model: str
    name: str
    version: Version
    model_name: ModelName
    # base_model: str
    # name: str
    # version: Version
    # model_name: ModelName
    raw_text: RawTextFile
    json_lines: JSONLinesFile
    dataset_dict: DatasetDictPath
    tokenizer: TokenizerPath
    model: ModelPath

    __fite_config = FITEConfig.load(
        Path(os.getenv("FITE_CONFIG", Path.cwd() / "pyproject.toml"))
    )

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def config(self) -> Model:
        return self.__fite_config.get_model(self.name)


@dataclasses.dataclass
class FileSystemDirectory(DataClassBase[str, Path]):
    """the file system directory maps the dataset name to the file system paths

    the root paths are
    - store/data -> raw text files and json lines files
    - store/datasets -> dataset dictionary files
    - store/models -> model files
    - store/tokenizer -> tokenizer files

    the file system requires the existence of a raw text file or a jsonl file
    formatted as follows:
    - raw text file: store/data/{name}-training-data.txt
    - jsonl file: store/data/{name}-training-data.jsonl

    the file system will create the following files:
    - dataset dictionary: store/datasets/{base_model}-{name}-{version}
    - tokenizer: store/tokenizer/{base_model}-{name}-{version}
    - model: store/models/{base_model}-{name}-{version}

    """

    base_model: str
    version: str
    store_path: Path = Path.cwd() / os.getenv("STORE_PATH", "store")

    def __post_init__(self):
        self.data_path = self.store_path / "data"

        dataset_names = {
            file.stem.rstrip("-training-data") for file in self.data_path.iterdir()
        }
        self.__fsd = dict(self._generate_file_map(dataset_names))
        """the file system directory maps the dataset name to the file system paths"""

    def _generate_file_map(
        self, dataset_names: set[str]
    ) -> Iterable[tuple[str, FileSystem]]:
        """generate the file system paths for each dataset name"""
        # unpack class attributes
        base_model, version = self.base_model, self.version
        data_path, store_path = self.data_path, self.store_path
        # iterate over the dataset names, to generate the a FileSystem Mapping
        for name in dataset_names:
            model_name = f"{base_model}-{name}-{version}"
            yield name, FileSystem(
                name=name,
                version=version,
                base_model=base_model,
                model_name=model_name,
                # the raw text and jsonl files are store in the data directory
                raw_text=data_path / f"{name}-training-data.txt",
                json_lines=data_path / f"{name}-training-data.jsonl",
                # the dataset dictionary, tokenizer, and model files are stored in the
                # respective directories behind the model name
                model=store_path / model_name / "model",
                tokenizer=store_path / model_name / "tokenizer",
                dataset_dict=store_path / model_name / "dataset",
            )

    def get(self, name: str) -> FileSystem:
        fs = self.__fsd.get(name)
        if fs is None:
            raise OSError(
                f"the dataset {name} does not exist in the file system directory"
            )

        return fs


@dataclasses.dataclass
class ResultRecord(
    DataClassBase[str, list[str], float, HyperParameters | HyperParameterStrategy]
):
    """ResultOutput is a named tuple that contains the output of the model."""

    model: str
    prompt_text: str
    generated_text: list[str]
    score: float
    strategy: str
    hyper_parameters: HyperParameterStrategy | HyperParameters


class StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return self.value

    def _generate_next_value_(name: str, *_: tuple) -> str:
        return name


class RegexEnum(StrEnum):
    def escape(self) -> str:
        return re.escape(str(self))

    @property
    def compile(self) -> re.Pattern[str]:
        return re.compile(self.escape())


class TokenEnum(RegexEnum):
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


class ActionStr(StrEnum):
    """ParserActions are the actions that the parser can take.
    >>> ParserActions.HELP
    'help'
    >>> ParserActions.STORE
    'store'
    """

    STORE = "store"
    CONST = "store_const"
    TRUE = "store_false"
    FALSE = "store_true"
    APPEND = "append"
    APPEND_CONST = "append_const"
    COUNT = "count"
    HELP = "help"
    VERSION = "version"


class TomlEncoder(toml.TomlEncoder):
    """TomlEncoder is a custom toml encoder to handle lists
    specifically how the toml.TomlEncoder handles lists:
    instead of:
    >>> toml.dumps({"a": ["1", "2", "3"]})
    'a = ["1", "2", "3"]'
    it will be:
    >>> toml.dumps({"a": ["1", "2", "3"]})
    'a = [
        "1",
        "2",
        "3",
    ]'
    """

    def dump_list(self, value_list: list) -> str:
        result = "["
        if value_list:
            result += "\n"
            for u in value_list:
                result += f"\t{self.dump_value(u)},\n"
        result += "]"
        return result
