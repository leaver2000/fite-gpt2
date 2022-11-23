from tokenizers.implementations import BertWordPieceTokenizer
from typing import TypedDict

from .util import get_raw_text_data


class TAFJsonLine(TypedDict):
    prompt: str
    completion: str


def main1():
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=False,
    )
    tokenizer.train(
        files=["/home/andrew/Projects/taf/taf-full.txt"],
        vocab_size=52_000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        wordpieces_prefix="##",
    )


from pathlib import Path


class TrainingData:
    def __init__(self, file_path: Path):
        self.file_path = file_path


import json
import textwrap

RAW_DATASET_PATH = Path("taf-full.json")
from typing import NamedTuple
import pandas as pd

StrPath = str | Path
import re

YEAR = 2021
MONTH = 12


class TAFDataset(NamedTuple):
    version: str
    description: str
    data: list[str]

    @classmethod
    def load(cls, file: StrPath = RAW_DATASET_PATH) -> "TAFDataset":
        if isinstance(file, str):
            file = Path(file)

        with file.open("r") as f:
            return TAFDataset(**json.load(f))

    def dump(self, file: StrPath = RAW_DATASET_PATH) -> None:
        if isinstance(file, str):
            file = Path(file)

        with file.open("w") as f:
            json.dump(self._asdict(), f, indent=4)

    def to_pandas(self) -> "pd.Series[str]":
        """
        - load the taf dataset into a pandas.Series
        - extract metadata from the TAF's
        - return a pandas.Series with the metadata as the index and the TAF as the value

        NOTE: TAF's dont have Year or Month information, so we will assume the current year and month

        ```
        >>> taf = TAFDataset.load()
        >>> taf.to_pandas().head(5)
        icao  amd    issue_time           from_valid_time      to_valid_time        max_temp  min_temp
        PABI  True   2021-12-14 07:04:00  2021-12-14 00:07:00  2021-12-15 01:02:00  3         -5          TAF AMD PABI 140740Z 1407/1512 11010G15KT 9999...
        KBLV  False  2021-12-16 08:00:00  2021-12-16 00:08:00  2021-12-17 01:04:00  2         -3          TAF KBLV 160800Z 1608/1714 27009KT 9999 SCT015...
        KDOV  False  2021-12-12 08:00:00  2021-12-12 00:08:00  2021-12-13 01:04:00  21         18         TAF KDOV 120800Z 1208/1314 22015G25KT 9999 BKN...
        PABI  True   2021-12-14 11:00:00  2021-12-14 01:01:00  2021-12-15 01:02:00  7         -5          TAF AMD PABI 141100Z 1411/1512 28015G25KT 9999...
        PASY  False  2021-12-11 06:00:00  2021-12-11 00:06:00  2021-12-12 01:02:00  5          2          TAF PASY 110600Z 1106/1212 16010G15KT 9999 OVC...
        Name: text, dtype: object
        ```
        """
        # create a pandas series from the data
        s = pd.Series(self.data, name="text")
        # regex pattern to extract the icao, amd, issue_time, valid_time_from, valid_time_to
        header_pattern = r"^TAF\s(?P<amd>AMD\s)?(?P<icao>[A-Z]{4})\s(?P<issue_time>\d{6})Z\s(?P<from_valid_time>\d{4})/(?P<to_valid_time>\d{4})"
        # extract header information
        df = s.str.extract(header_pattern, expand=True)
        # extract temperature information at th end of each taf
        temperature_pattern = r"TX(M?\d{2})/\d{4}Z\sTN(M?\d{2})/\d{4}Z$"
        df[["max_temp", "min_temp"]] = (
            s.str.extract(temperature_pattern, expand=True)
            .stack()
            .str.replace("M", "-", regex=False)
            .unstack()
            .astype(int)
        )
        df["amd"] = df["amd"].notna()

        df["issue_time"] = pd.to_datetime(
            f"{YEAR}{MONTH}" + df.issue_time, format="%Y%m%d%H%M%S"
        )
        df["from_valid_time"] = pd.to_datetime(
            f"{YEAR}{MONTH}" + df.from_valid_time, format="%Y%m%d%H%M"
        )
        df["to_valid_time"] = pd.to_datetime(
            f"{YEAR}{MONTH}" + df.to_valid_time, format="%Y%m%d%H%M"
        )
        columns = [
            "icao",
            "amd",
            "issue_time",
            "from_valid_time",
            "to_valid_time",
            "max_temp",
            "min_temp",
        ]
        s.index = df[columns].pipe(pd.MultiIndex.from_frame)

        return s

    def to_lines(self) -> "pd.Series[str]":
        """"""
        return self.to_pandas().head(5).str.split("\n").explode().str.strip().dropna()


def main():
    df = TAFDataset.load()
    lines = df.to_lines()
    print(lines.reset_index())
    # print(df.dump("taf.jsonl"))


if __name__ == "__main__":
    main()
