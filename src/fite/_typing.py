from pathlib import Path
from typing import Any, Optional, TypeAlias, TypedDict

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


ModelConfig = TypedDict(
    "ModelConfig",
    {
        "base-model": str,  # gpt2
        "dataset": str,  # taf
        "version": str,  # base1 | base1.2021-09-01
        "model-name": str,  # gpt2-taf-base1 | gpt2-taf-base1.2021-09-01
        "description": str,
        "metadata-pattern": Optional[str],
        "additional-tokens": list[str],
        "additional-special-tokens": list[str],
        "prompt-examples": list[str],
    },
)

ProjectConfig = TypedDict(
    "ProjectConfig", {"root-path": str, "models": list[ModelConfig]}
)


class Project(TypedDict):
    name: str
    version: str
    description: str
    long_description: str
    long_description_content_type: str
    url: str


class Tools(TypedDict):
    fite: ProjectConfig


_PyProject = TypedDict(
    "_PyProject",
    {"build-system": TOMLDict, "project": Project, "tool": Tools},
)


class PyProjectTOML(_PyProject):
    """the pyproject.toml file is the configuration file for the fite package
    PEP 621 - Storing project metadata in pyproject.toml

    https://www.python.org/dev/peps/pep-0621/

    https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/

    """

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PyProjectTOML":
        if not path:
            path = "pyproject.toml"

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise ValueError(f"Config file not found: {path}")

        with path.open("r") as f:
            project = toml.load(f)

        return cls(**project)  # type: ignore

    def dump(self, path: str | Path = "pyproject.toml") -> None:  # type: ignore
        if isinstance(path, str):
            path = Path(path)

        with path.open("w") as f:
            toml.dump(self, f, encoder=TomlEncoder())


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
