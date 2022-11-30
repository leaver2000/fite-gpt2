from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import toml

__all__ = [
    "StrPath",
    "ModelName",
    "Version",
    "RawTextFile",
    "JSONLinesFile",
    "DatasetDictPath",
    "TokenizerPath",
    "ModelPath",
    "PyProjectTOML",
    "Model",
    "Project",
    "PyProjectTOML",
]


StrPath: TypeAlias = str | Path

ModelName: TypeAlias = str
Version: TypeAlias = str
RawTextFile: TypeAlias = Path
JSONLinesFile: TypeAlias = Path
DatasetDictPath: TypeAlias = Path
TokenizerPath: TypeAlias = Path
ModelPath: TypeAlias = Path

TOMLDict = dict[str, list | dict | str | int | float | bool | None | Any]


class Model(TypedDict):
    name: str
    prompt_examples: list[str]


class Project(TypedDict):
    name: str
    version: str
    description: str
    long_description: str
    long_description_content_type: str
    url: str
    models: list[Model]


_PyProject = TypedDict(
    "_PyProject",
    {"build-system": TOMLDict, "project": Project, "tool": TOMLDict},
)


class PyProjectTOML(_PyProject):
    @classmethod
    def load(cls, path: str | Path = "pyproject.toml") -> "PyProjectTOML":
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Config file not found: {path}")
        with path.open("r") as f:
            return cls(**toml.load(f))  # type: ignore
