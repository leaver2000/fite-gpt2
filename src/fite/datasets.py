import dataclasses
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypedDict

import datasets
import pandas as pd
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.dataset_dict import DatasetDict

from .util import FileSystem, SpecialTokens

__all__ = ["Dataset", "TextFile"]

TEMPERATURE_GROUP_PATTERN = (
    r"\sTX?(?P<max_temp>M?\d{2})\/\d{4}Z\sTN?(?P<min_temp>M?\d{2})\/\d{4}Z$"
)
CHANGE_GROUP_PATTERN = r"\s(?=BECMG|TEMPO)"
SPLIT_PATTERN = "\n\n###\n\n"
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


# def _generate_datums(rows: list[dict[str, str | dict[str, str]]]) -> Iterable[Datum]:
#     """
#     iterate over the rows splitting each word, the split text is used to create the prompt and completion.
#     the __text__ is popped from each dict to create the prompt and completion.
#     the remaining dict is used as the metadata.
#     """
#     for row in rows:
#         prompt = ""
#         text_list = row.pop("__text__").split()  # type: ignore
#         # row = toml.dumps(row)
#         for i, text in enumerate(text_list):
#             prompt += f"{text} "

#             yield Datum(
#                 metadata=row,
#                 prompt=prompt,
#                 completion=" ".join(text_list[i + 1 :]),
#             )


MetadataCallback = Callable[[pd.Series], pd.Series | pd.DataFrame]


class Dataset(ArrowDataset):
    """A dataset dictionary"""

    def __getitem__(self, __key: Literal["train", "test", "validate"]) -> ArrowDataset:
        return super().__getitem__(__key)  # type: ignore

    @classmethod
    def from_json(
        cls, path: Path | str, split: float = 0.2, shuffle: bool = True
    ) -> "DatasetDict":
        """Create a dataset dictionary from a jsonl file."""
        if isinstance(path, Path):
            path = str(path)
        return ArrowDataset.from_json(str(path), features=FEATURES).train_test_split(  # type: ignore
            test_size=split, shuffle=shuffle
        )




@dataclasses.dataclass
class TextFile:
    """A class to handle raw text files"""

    fs: FileSystem
    split_pattern: str = r"\n+###+\n+"
    split: float = 0.2
    shuffle: bool = True
    metadata_handler: Callable[["pd.Series[str]"], pd.DataFrame] | None = None
    extract_pattern: str | None = None

    @staticmethod
    def _metar_handler(s: "pd.Series[str]") -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _taf_handler(s: "pd.Series[str]") -> pd.DataFrame:
        return s.str.extract(
            r"\sTX?(?P<maximum_temperature>M?\d{2})\/\d{4}Z\sTN?(?P<minimum_temperature>M?\d{2})\/\d{4}Z$"
        )

    __handlers = {"taf": _taf_handler, "metar": _metar_handler}

    def extract(self, s: "pd.Series[str]") -> pd.DataFrame:
        if not self.extract_pattern:
            raise ValueError("escaped_extraction_pattern must be provided")
        return s.str.extract(self.extract_pattern)

    def __post_init__(self):
        sep = re.compile(self.split_pattern)
        with self.fs.raw_text.open("r") as f:
            s = pd.Series(sep.split(f.read()), name="text").str.strip()
        if self.metadata_handler and self.extract_pattern:
            import warnings

            warnings.warn(
                "metadata_handler and extract_pattern are both provided. extract_pattern will be ignored."
            )
        metadata_handler = (
            self.metadata_handler if self.metadata_handler else self._metadata_handlers
        )

        self._frame = pd.DataFrame(
            s.pipe(metadata_handler).pipe(self._generate_json_lines),
            columns=["metadata", "prompt", "completion"],
        ).drop_duplicates(ignore_index=True)

    def _metadata_handlers(self, __s: "pd.Series[str]") -> pd.DataFrame:
        handler = (
            self.extract if self.extract_pattern else self.__handlers.get(self.fs.name)
        )
        if not handler:
            import warnings

            warnings.warn(f"No metadata handler detected for {self.fs.name}")
            return __s.to_frame()
        return __s.pipe(handler).join(__s)

    @staticmethod
    def _generate_json_lines(df: pd.DataFrame) -> Iterable[tuple[str, str, str]]:
        """
        iterate over the rows splitting each word, the split text is used to create the prompt and completion.
        the text is popped from each dict to create the prompt and completion.
        the remaining dict is used as the metadata.
        """
        df["text"] = df.text.str.strip().str.split()
        df.columns = df.columns.str.replace("_", "-")

        for _, metadata in df.iterrows():
            prompt = SpecialTokens.bos_token
            text_list = metadata.pop("text")
            metadata = f"{SpecialTokens.metadata}\n" + "\n".join(
                f"{k} = {v}" for k, v in metadata.items()
            )

            for i, text in enumerate(text_list):
                prompt += f"{text} "
                completion = " ".join(text_list[i:]) + SpecialTokens.eos_token
                yield metadata, prompt, completion

    def to_dataset(
        self, test_size: float = 0.2, shuffle: bool = True, **kwargs
    ) -> DatasetDict:
        return ArrowDataset.from_pandas(self._frame).train_test_split(
            test_size=test_size, shuffle=shuffle, **kwargs
        )

    def save_to_disk(self) -> None:
        self._frame.to_json(self.fs.json_lines, orient="records", lines=True)
