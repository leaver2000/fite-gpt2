import dataclasses
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypedDict

import datasets
import pandas as pd
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.dataset_dict import DatasetDict

from ..util import SpecialTokens

__all__ = ["Dataset", "TextFile"]

FEATURES = datasets.Features(
    {
        "metadata": datasets.Value("string"),
        "prompt": datasets.Value("string"),
        "completion": datasets.Value("string"),
    }
)


class Datum(TypedDict):
    metadata: Any
    prompt: str
    completion: str


MetadataCallback = Callable[[pd.Series], pd.Series | pd.DataFrame]


class Dataset(ArrowDataset):
    """A dataset dictionary"""

    def __getitem__(self, __key: Literal["train", "test", "validate"]) -> ArrowDataset:
        return super().__getitem__(__key)  # type: ignore

    @classmethod
    def from_json_lines(
        cls,
        path: Path | str,
        split: float = 0.2,
        shuffle: bool = True,
        features: datasets.Features | None = None,
    ) -> "DatasetDict":
        """Create a dataset dictionary from a jsonl file."""

        return (
            super()
            .from_json(str(path), features=features)
            .train_test_split(test_size=split, shuffle=shuffle)  # type: ignore
        )


@dataclasses.dataclass
class TextFile:
    """A class to handle raw text files

    the split_pattern should match the pattern used to segment elements in the fs.raw_text_file
    >>> text = TextFile("hello world\nspam eggs", split_pattern=" ")
    >>> text.to_dataframe()
         prompt completion
    0     hello      world
    >>> text.to_jsonl("hello_world.jsonl")
    >>> text.to_dataset()
    DatasetDict({
        train: Dataset({
            features: ['prompt', 'completion'],
            num_rows: 1
        })
        test: Dataset({
            features: ['prompt', 'completion'],
            num_rows: 0
        })
    """

    text: str
    metadata_pattern: str | None = None
    split_pattern: str = r"\n+#+\n+"

    def __post_init__(self):
        s = pd.Series(re.split(self.split_pattern, self.text), name="text").str.strip()

        self.__frame = pd.DataFrame(
            s.pipe(self._extract_metadata).pipe(self._generate_json_lines),
            columns=["prompt", "completion"],
        ).drop_duplicates(ignore_index=True)

    @classmethod
    def from_file(cls, file: Path, metadata_pattern: str | None = None) -> "TextFile":
        """
        >>> fs = FileSystem(CONFIG)
        >>> with fs.raw_text_file.open("w") as f:
        ...     f.write("this is a test\\n###\\nthis is another test")
        >>> text = TextFile.from_file(fs.raw_text_file, split_pattern="\\n###\\n")
        >>> text.to_dataframe()
           prompt          completion
        0  this is a test  this is another test
        """

        with file.open("r") as f:
            return cls(
                f.read(),
                metadata_pattern=metadata_pattern,  # fs.config.get("metadata-pattern", None)
            )

    def _extract_metadata(self, __s: "pd.Series[str]") -> pd.DataFrame:
        # pattern = self.metadata_pattern
        # TODO: use the metadata pattern to extract metadata from the text
        return __s.str.extract(r"(?:(TXM?\d{2}).+(TNM?\d{2}))").join(__s)


    @staticmethod
    def _generate_json_lines(
        df: pd.DataFrame,
    ) -> Iterable[tuple[str, str, str] | tuple[str, str]]:
        """
        iterate over the rows splitting each word, the split text is used to create the prompt and completion.
        the text is popped from each dict to create the prompt and completion.
        the remaining dict is used as the metadata.
        """
        template = "TAF [{header}] {text}"
        head, *_ = template.split("[")
        header = df.drop(columns="text").apply(" ".join, axis=1).rename("header")

        text = (
            df.text.str.replace(head, "")
            .to_frame()
            .join(header)
            .apply(lambda s: template.format(**s), axis=1)
        )
        for text_list in text.str.strip().str.split():
            prompt = SpecialTokens.bos_token
            for i, text in enumerate(text_list):
                prompt += f"{text} "
                if completion := " ".join(text_list[i + 1 :]):
                    yield prompt, completion + SpecialTokens.eos_token

    def to_dataframe(self) -> pd.DataFrame:
        return self.__frame

    def to_dataset_dict(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        features: datasets.Features | None = None,
        **kwargs,
    ) -> DatasetDict:
        return ArrowDataset.from_pandas(
            self.to_dataframe(), features=features
        ).train_test_split(test_size=test_size, shuffle=shuffle, **kwargs)

    def to_jsonl(self, path: Path) -> None:
        self.to_dataframe().to_json(path, orient="records", lines=True)
