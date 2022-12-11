import argparse

import dataclasses
import re
from pathlib import Path
from typing import Literal, overload, Optional
import attrs
from datasets import Features, Value
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config

from .._typing import (
    DatasetDictPath,
    JSONLinesFile,
    ModelConfig as FileSystemConfig,
    ModelPath,
    ProjectConfig,
    PyProjectTOML,
    RawTextFile,
    TokenizerPath,
    Version,
)
from ..pipeline import Pipeline, HyperParameterStrategy
from ..util import (
    CONSTANTS,
    DEFAULT_DEVICE,
    DataclassBase,
    GPT2BaseModels,
    StrEnum,
    ActivationFunctions,
)

__all__ = ["FileSystem", "FileSystemDirectory", "Namespace"]


class Namespace(argparse.Namespace):
    filesystem: str
    all: bool
    out_dir: Optional[str]
    verbose: bool = False

    def iterstrategies(self):
        if self.all:
            yield from HyperParameterStrategy
        else:
            for strategy in HyperParameterStrategy:
                if getattr(self, strategy.name.lower()):
                    yield strategy

    @classmethod
    def with_strategies(cls, parser: argparse.ArgumentParser) -> "Namespace":
        parser.add_argument("--all", action="store_true")
        for strategy in HyperParameterStrategy:
            parser.add_argument(f"--{strategy.name.lower()}", action="store_true")
        return cls()


@dataclasses.dataclass
class FileSystem(DataclassBase[str, Path, FileSystemConfig]):
    root_path: Path
    raw_text_file: RawTextFile
    json_lines_file: JSONLinesFile
    dataset_dict_path: DatasetDictPath
    tokenizer_path: TokenizerPath
    model_path: ModelPath
    config: FileSystemConfig = attrs.field(on_setattr=attrs.setters.frozen)

    def __post_init__(self):
        # if the base-model is gpt2 or gpt2-medium then it is not a local model
        # if the base-model is gpt2-taf-base1 then it is a local model
        # from the perspective of the filesystem
        self.model_is_local = (
            self.config["base-model"] not in GPT2BaseModels.list_values()
        )

    @property
    def model_name(self) -> str:
        return self.config["model-name"]

    @property
    def base_model(self) -> Path | str:
        # base_model = self.config["base-model"]
        return self.config["base-model"]

    @property
    def dataset(self) -> str:
        return self.config["dataset"]

    @property
    def version(self) -> Version:
        return self.config["version"]

    @property
    def metadata_pattern(self) -> str | None:
        pattern = self.config.get("metadata-pattern")
        if pattern:
            return re.escape(pattern)

    def get_model(self, **kwargs) -> GPT2LMHeadModel:
        """fine-tuned model"""
        return GPT2LMHeadModel.from_pretrained(self.model_path, **kwargs)  # type: ignore

    def get_tokenizer(self, **kwargs) -> GPT2TokenizerFast:
        """fine-tuned tokenizer"""
        # You're using a GPT2TokenizerFast tokenizer.
        # Please note that with a fast tokenizer, using the `__call__` method is faster
        # than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
        return GPT2TokenizerFast.from_pretrained(self.tokenizer_path, **kwargs)

    def get_pipeline(self, **kwargs) -> Pipeline:
        return Pipeline(
            model=self.get_model(),
            tokenizer=self.get_tokenizer(),
            device=DEFAULT_DEVICE,
            max_length=CONSTANTS.MAX_LENGTH,
            num_return_sequences=1,
            **kwargs,
        )

    def get_features(self) -> Features:
        features = {
            "prompt": Value("string"),
            "completion": Value("string"),
        }
        return Features(features)

    @overload
    def get_pretrained(self, kind: Literal["MODEL"], **kwargs) -> GPT2LMHeadModel:
        ...

    @overload
    def get_pretrained(self, kind: Literal["TOKENIZER"], **kwargs) -> GPT2TokenizerFast:
        ...

    def get_pretrained(
        self, kind: Literal["MODEL", "TOKENIZER"], **kwargs
    ) -> GPT2LMHeadModel | GPT2TokenizerFast:
        """
        the function is used to load the pretrained model and tokenizer
        currently only supports models & tokenizers from huggingface or the local file system
        """
        # get the base model
        base_model = self.config["base-model"]
        # if the model is local, then the path is the root path + base model + kind
        pretrained_model_name_or_path = str(
            self.root_path / base_model / kind.lower()
            if self.model_is_local
            else base_model
        )

        if kind == "MODEL":
            if self.model_is_local:
                # local model loads from the previous model config
                config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
            else:
                config = GPT2Config(
                    activation_function=ActivationFunctions.gelu_new,  # ["relu", "silu", "gelu", "tanh", "gelu_new"]
                    layer_norm_eps=1e-05,
                )

            return GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
            )  # type: ignore

        elif kind == "TOKENIZER":
            return GPT2TokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                num_labels=2,
                do_basic_tokenize=False,
                **kwargs,
            )

        else:
            raise ValueError(f"kind must be model or tokenizer, got {kind}")

    @overload
    def get(self):
        ...

    @property
    def get(self):
        """calls to the underlying FileSystemConfig dictionary"""
        return self.config.get


@dataclasses.dataclass
class FileSystemDirectory(DataclassBase[str, Path]):
    """the file system directory maps the dataset name to the file system paths

    model naming convention: {base-model}-{dataset}-{version}
    ie. gpt2-taf-base1 or gpt2-taf-base1.2022-01-01

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

    project_config: ProjectConfig

    def __post_init__(self):
        project_config = self.project_config
        root_path = Path(project_config["root-path"])
        self.__fsd: dict[str, FileSystem] = {}

        for config in project_config["models"]:
            # access the model name from the config file
            model_name = config["model-name"]
            model_root = root_path / model_name

            if not model_root.exists():
                model_root.mkdir(parents=True)

            self.__fsd[model_name] = FileSystem(
                root_path=root_path,
                config=config,
                raw_text_file=model_root / FileNames.RAW_TEXT,
                json_lines_file=model_root / FileNames.JSON_LINES,
                dataset_dict_path=model_root / FileNames.DATASET_DICT,
                tokenizer_path=model_root / FileNames.TOKENIZER,
                model_path=model_root / FileNames.MODEL,
            )

    @classmethod
    def from_pyproject(cls, py_project: PyProjectTOML) -> "FileSystemDirectory":
        return cls(project_config=py_project["tool"]["fite"])

    @classmethod
    def load_from_pyproject(
        cls, path: str | Path | None = None
    ) -> "FileSystemDirectory":
        return cls.from_pyproject(PyProjectTOML.load(path))

    def get(self, name: str) -> FileSystem:
        fs = self.__fsd.get(name)
        if fs is None:
            raise OSError(
                f"the dataset {name} {self.__fsd} does not exist in the file system directory"
            )

        return fs

    def get_pipeline(self, name: str, **kwargs) -> Pipeline:
        return self.get(name).get_pipeline(**kwargs)

    def list_models(self) -> list[FileSystem]:
        return list(self.__fsd.values())

    def list_model_names(self) -> list[str]:
        return list(self.__fsd.keys())


class FileNames(StrEnum):
    """the file names for the file system"""

    RAW_TEXT = "training-data.txt"
    JSON_LINES = "training-data.jsonl"
    DATASET_DICT = "dataset"
    TOKENIZER = "tokenizer"
    MODEL = "model"
