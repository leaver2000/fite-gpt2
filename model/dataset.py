import json
import random
import dataclasses
from pathlib import Path
from datetime import datetime
from typing import NamedTuple, TypedDict, TypeAlias, Iterable

import pandas as pd
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset

from .util import RegexPatterns, SpecialTokens


__all__ = ["TAFDataset", "JSONLines"]
YEAR = 2021
MONTH = 12

RAW_DATASET_PATH = Path("taf-full.json")
TAFSeries: TypeAlias = "pd.Series[str]"
StrPath = str | Path


class Line(TypedDict):
    prompt: str
    completion: str


class LineWithMeta(Line):
    icao: str
    amd: bool
    issue_time: datetime
    from_valid_time: datetime
    to_valid_time: datetime
    max_temp: int
    min_temp: int
    prompt: str
    completion: str


class JSONLines(NamedTuple):
    lines: tuple[Line, ...]
    metadata: pd.MultiIndex = dataclasses.field(repr=False)
    __index = [
        "icao",
        "amd",
        "issue_time",
        "from_valid_time",
        "to_valid_time",
        "max_temp",
        "min_temp",
    ]
    __columns = ["prompt", "completion"]

    @classmethod
    def load(
        cls,
        file: StrPath,
        strip_temps: bool = True,
        drop_wnd_aft_rmks: bool = True,
        pad_prompt: bool = True,
        pad_completion: bool = True,
    ) -> "JSONLines":

        df = pd.read_json(file, lines=True, convert_dates=False)
        if strip_temps:
            # the temperature group when encoding line by line is not needed
            df["completion"] = df.completion.str.replace(
                RegexPatterns.TEMPERATURE_GROUP, "", regex=True
            )

        if drop_wnd_aft_rmks:
            # there are not enough sample with wind after remarks to be useful
            drop_index = df.index[
                df.prompt.str.contains("WND") | df.completion.str.contains("WND")
            ]
            df = df.drop(drop_index)
        if pad_completion:
            df["completion"] = df.completion + SpecialTokens.eos_token

        if pad_prompt:
            df["prompt"] = SpecialTokens.bos_token + df.prompt
        # df
        return JSONLines(
            lines=tuple(df[cls.__columns].to_dict("records")),  # type: ignore
            metadata=pd.MultiIndex.from_frame(df[cls.__index]),
        )  # type: ignore

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.lines, index=self.metadata)

    def to_dataset_dict(
        self, test_size: float = 0.2, random_state: int = 0
    ) -> DatasetDict:
        """
        - split the dataset into train and test as a dataset dict

        ```
        >>> JSONLines.load("store/taf.jsonl").to_dataset_dict(
            test_size=0.2,
            random_state=8,
        )

        DatasetDict({
            train: Dataset({
                features: ['prompt', 'completion'],
                num_rows: 11302
            })
            validation: Dataset({
                features: ['prompt', 'completion'],
                num_rows: 2825
            })
        })
        ```
        """
        df = self.to_frame().reset_index()[self.__columns]

        train = df.sample(frac=1 - test_size, random_state=random_state)
        validation = df.drop(train.index)

        return DatasetDict(
            train=Dataset.from_pandas(train, preserve_index=False),
            validation=Dataset.from_pandas(validation, preserve_index=False),
        )


class TAFDataset(NamedTuple):
    # version: str
    # description: str
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

    def to_series(self) -> TAFSeries:
        """
        - load the taf dataset into a pandas.Series
        - extract metadata from the TAF's
        - return a pandas.Series with the metadata as the index and the TAF as the value

        NOTE: TAF's dont have Year or Month information, so we will assume the current year and month

        ```
        >>> taf = TAFDataset.load()
        >>> taf.to_series().head(5)
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
        s.index = pd.MultiIndex.from_frame(df[columns])
        return s

    @staticmethod
    def _generate_lines(split_lines: Iterable[list[str]]):
        for line in split_lines:
            if line == [""]:
                yield Line(prompt="", completion="")
                continue
            elif line[0] == "TAF":
                start_split = 3  # [TAF KBLV 160800Z]
                if line[:2] == ["TAF", "AMD"]:
                    start_split = 4  # [TAF AMD KBLV 160800Z]

            elif line[0] in ("BECMG", "TEMPO"):
                start_split = 2  # BECMG 1613/1614 | TEMPO 1409/1414
            else:
                raise ValueError(f"Unknown TAF line type: {line[0]}")

            index = random.randint(start_split, len(line) - 1)

            yield Line(prompt=" ".join(line[:index]), completion=" ".join(line[index:]))

    def to_json_lines(
        self,
        split_each_line: bool = True,
    ) -> JSONLines:
        """"""
        s = self.to_series()
        if split_each_line:
            s = s.str.split("\n").explode()

        lines = tuple(s.str.split(" ").pipe(self._generate_lines))
        return JSONLines(lines=lines, metadata=s.index)  # type: ignore

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = 0
    ) -> tuple[TAFSeries, TAFSeries]:
        s = self.to_series()
        # split the training data as a fraction of test_size
        train = s.sample(frac=1 - test_size, random_state=random_state)
        # drop the training data from the original dataframe to create the test data
        test = s.drop(train.index)
        # return the train and test dataframes
        return train, test


def _prepare_json_lines(by=1_000):
    """
    artificially increase the size of the dataset by iterating and randomizing the split
    """
    taf = TAFDataset.load()

    df = (
        pd.concat(taf.to_json_lines().to_frame() for _ in range(by))
        .drop_duplicates()
        .reset_index()
    )

    assert all(df.value_counts() == 1)
    # return df
    time_cols = ["issue_time", "from_valid_time", "to_valid_time"]
    df[time_cols] = df[time_cols].stack().dt.strftime("%Y-%m-%dT%H:%M:%SZ").unstack()

    with open("taf.jsonl", "w") as f:
        for record in df.to_dict(orient="records"):
            json.dump(record, f)
            f.write("\n")


def main():

    ds = JSONLines.load("store/training-taf-data.jsonl").to_dataset_dict(
        test_size=0.2,
        random_state=8,
    )
    print(ds)


if __name__ == "__main__":
    main()
