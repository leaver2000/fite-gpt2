import dataclasses
from pathlib import Path

import attrs


from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .._typing import (
    DatasetDictPath,
    JSONLinesFile,
    ModelConfig,
    ModelPath,
    ProjectConfig,
    PyProjectTOML,
    RawTextFile,
    TokenizerPath,
    Version,
)
from ..pipeline import Pipeline
from ..util import CONSTANTS, DEFAULT_DEVICE, DataclassBase


@dataclasses.dataclass
class FileSystem(DataclassBase[str, Path, ModelConfig]):
    raw_text_file: RawTextFile
    json_lines_file: JSONLinesFile
    dataset_dict_path: DatasetDictPath
    tokenizer_path: TokenizerPath
    model_path: ModelPath
    config: ModelConfig = attrs.field(on_setattr=attrs.setters.frozen)

    def get_model(self, **kwargs) -> GPT2LMHeadModel:
        return GPT2LMHeadModel.from_pretrained(self.model_path, **kwargs).to(DEFAULT_DEVICE)  # type: ignore

    def get_tokenizer(self, **kwargs) -> GPT2TokenizerFast:
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

    @property
    def model_name(self) -> str:
        return self.config["model-name"]

    @property
    def base_model(self) -> str:
        return self.config["base-model"]

    @property
    def dataset(self) -> str:
        return self.config["dataset"]

    @property
    def version(self) -> Version:
        return self.config["version"]


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
            # if it does not exist then use the template
            # if not model_name:
            expected_model_name, *_ = "-".join(
                config[key] for key in ("base-model", "dataset", "version")
            ).split(".")
            assert model_name == expected_model_name

            model_root = root_path / model_name
            if not model_root.exists():
                model_root.mkdir(parents=True)
            self.__fsd[model_name] = FileSystem(
                config=config,
                raw_text_file=model_root / "training-data.txt",
                json_lines_file=model_root / "training-data.jsonl",
                dataset_dict_path=model_root / "dataset",
                tokenizer_path=model_root / "tokenizer",
                model_path=model_root / "model",
            )

    @classmethod
    def from_pyproject(cls, py_project: PyProjectTOML) -> "FileSystemDirectory":
        return cls(project_config=py_project["tool"]["fite"])

    @classmethod
    def load_from_pyproject(
        cls, path: str | Path | None = None
    ) -> "FileSystemDirectory":
        return cls.from_pyproject(PyProjectTOML.load(path))

    @staticmethod
    def _format_model_name(config: ModelConfig) -> str:
        return f"{config['base-model']}-{config['dataset']}-{config['dataset']}"

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
