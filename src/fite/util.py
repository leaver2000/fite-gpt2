import dataclasses
import enum
import os
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Generic, Optional, Union

import attrs

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

import torch
from transformers import AddedToken, GPT2LMHeadModel, GPT2TokenizerFast

from .enum import RegexEnum, StrEnum
from .pipeline import CodePredictionPipeline, HyperParameters, HyperParameterStrategy
from .typing import (
    DatasetDictPath,
    JSONLinesFile,
    ModelConfig,
    ModelPath,
    PyProjectTOML,
    RawTextFile,
    TokenizerPath,
    Version,
)

__all__ = [
    "FileSystem",
    "FileSystemDirectory",
    "ResultRecord",
    "SpecialTokens",
    "ActionStr",
    "CONSTANTS",
]
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

Ts = TypeVarTuple("Ts")


class CONSTANTS(enum.IntEnum):
    BATCH_SIZE = os.getenv("BATCH_SIZE", 8)
    MAX_LENGTH = os.getenv("MAX_LENGTH", 256)
    NUM_PROC = os.getenv("NUM_PROC", 4)


@dataclasses.dataclass
class DataclassBase(Generic[Unpack[Ts]]):
    def to_dict(self) -> dict[str, Union[Unpack[Ts]]]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FileSystem(DataclassBase[str, Path, ModelConfig]):
    base_model: str
    name: str
    version: str

    raw_text_file: RawTextFile
    json_lines_file: JSONLinesFile
    dataset_dict_path: DatasetDictPath
    tokenizer_path: TokenizerPath
    model_path: ModelPath
    config: ModelConfig = attrs.field(on_setattr=attrs.setters.frozen)

    @property
    def model_name(self) -> str:
        return f"{self.base_model}-{self.name}-{self.version}"

    def get_model(self, **kwargs) -> GPT2LMHeadModel:
        return GPT2LMHeadModel.from_pretrained(self.model_path, **kwargs).to(DEFAULT_DEVICE)  # type: ignore

    def get_tokenizer(self, **kwargs) -> GPT2TokenizerFast:
        # You're using a GPT2TokenizerFast tokenizer.
        # Please note that with a fast tokenizer, using the `__call__` method is faster
        # than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
        return GPT2TokenizerFast.from_pretrained(self.tokenizer_path, **kwargs)

    def get_pipeline(self, **kwargs) -> CodePredictionPipeline:
        return CodePredictionPipeline(
            model=self.get_model(),
            tokenizer=self.get_tokenizer(),
            device=DEFAULT_DEVICE,
            max_length=CONSTANTS.MAX_LENGTH,
            num_return_sequences=1,
            **kwargs,
        )


@dataclasses.dataclass
class FileSystemDirectory(DataclassBase[str, Path]):
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

    version: Version = "base"
    model_run: Optional[datetime] = dataclasses.field(repr=False, default=None)
    py_project: PyProjectTOML = dataclasses.field(
        init=False, repr=False, default_factory=PyProjectTOML.load
    )

    def __post_init__(self):
        config = self.py_project["tool"]["fite"]
        root_path = Path(config["root-path"])

        if self.model_run:
            self.version = self.model_run.strftime("v%Y.%m.%d.%H")

        self.__fsd: dict[str, FileSystem] = {}

        for model in config["models"]:
            model_root = root_path / model["model-name"]

            if not model_root.exists():
                model_root.mkdir(parents=True)
            # TODO: future versions will use a different model name scheme
            # gpt2-taf-v2021.09.01.12
            # in which case the base model will need to inherit from
            # gtp2-taf-base1 and this split wont work.
            base_model, name, version = model["model-name"].split("-")

            self.__fsd[name] = FileSystem(
                config=model,
                name=name,
                base_model=base_model,
                version=version,
                raw_text_file=model_root / "training-data.txt",
                json_lines_file=model_root / "training-data.jsonl",
                dataset_dict_path=model_root / "dataset",
                tokenizer_path=model_root / "tokenizer",
                model_path=model_root / "model",
            )

    def get(self, name: str) -> FileSystem:
        fs = self.__fsd.get(name)
        if fs is None:
            raise OSError(
                f"the dataset {name} {self.__fsd} does not exist in the file system directory"
            )

        return fs

    def get_pipeline(self, name: str, **kwargs) -> CodePredictionPipeline:
        return self.get(name).get_pipeline(**kwargs)

    def list_models(self) -> list[FileSystem]:
        return list(self.__fsd.values())

    def list_model_names(self) -> list[str]:
        return list(self.__fsd.keys())


@dataclasses.dataclass
class ResultRecord(
    DataclassBase[str, list[str], float, HyperParameters | HyperParameterStrategy]
):
    """ResultOutput is a named tuple that contains the output of the model."""

    model: str
    prompt_text: str
    generated_text: list[str]
    strategy: str
    hyper_parameters: HyperParameterStrategy | HyperParameters

    def evaluate(self):
        self.score = 0.0

        return self


AdditionalSpecialTokens = list[str] | list[AddedToken] | list[str | AddedToken]
SpecialTokenDict = dict[
    str, AddedToken | str | AdditionalSpecialTokens
]  # list[str | AddedToken] | list[str] | list[AddedToken]]


class TokenEnum(RegexEnum):
    value: str

    @classmethod
    def to_dict(cls) -> SpecialTokenDict:
        return {k: v.value for k, v in cls._member_map_.items()}

    @classmethod
    def with_additional(cls, tokens: AdditionalSpecialTokens) -> SpecialTokenDict:
        special_tokens = cls.to_dict().copy()
        if tokens:
            special_tokens["additional_special_tokens"] = tokens
        return special_tokens

    @classmethod
    def select(cls, *names) -> SpecialTokenDict:
        return {name: cls[name].value for name in names}


class SpecialTokens(TokenEnum):
    """Special tokens that can be added to a tokenizer"""

    bos_token = "[bos]"
    eos_token = "[eos]"
    pad_token = "[pad]"
    # cls_token = "<|cls|>"
    # pad_token = "<|pad|>"
    # unk_token = "<|unk|>"
    metadata = "[metadata]"


class ActivationFunctions(StrEnum):
    """Activation functions"""

    relu = enum.auto()
    silu = enum.auto()
    gelu = enum.auto()
    gleu_new = enum.auto()
    tanh = enum.auto()


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


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield
