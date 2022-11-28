import json
from pathlib import Path
from typing import Mapping, TypedDict, Iterable, Literal

import pandas as pd
import datasets
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset

from .util import SpecialTokens
from .typing import StrPath, GPT2TokenizerType

__all__ = ["dataset_dict"]


FEATURES = datasets.Features(
    {
        "TX_TN": datasets.Value("string"),
        "prompt": datasets.Value("string"),
        "completion": datasets.Value("string"),
    }
)
TEMPERATURE_GROUP_PATTERN = (
    r"\sTX?(?P<max_temp>M?\d{2})\/\d{4}Z\sTN?(?P<min_temp>M?\d{2})\/\d{4}Z$"
)
CHANGE_GROUP_PATTERN = r"\s(?=BECMG|TEMPO)"


class TafDatum(TypedDict):
    TX_TN: tuple[int, int]
    prompt: str
    completion: str


def _make_json_lines(text_file: StrPath, json_lines_file: StrPath) -> None:
    if isinstance(json_lines_file, str):
        json_lines_file = Path(json_lines_file)

    if isinstance(text_file, str):
        text_file = Path(text_file)

    with text_file.open("r") as f:
        taf_series = pd.Series(f.read().split("\n\n###\n\n"), name="text").str.strip()

    taf_series = (
        # extract the temperature groups
        taf_series.str.extract(TEMPERATURE_GROUP_PATTERN)
        # stack the temperature groups into a single column
        .stack()
        # so that we can replace M with - and convert to int
        .str.replace("M", "-")
        .unstack()
        .astype(int)
        # join the temperature groups with the original text
        .join(taf_series)
        # set the temperature groups as the index
        .set_index(["max_temp", "min_temp"])
        # convert the DataFrame back to a single text column
        .text
    )

    def generate(
        taf_mapping: Mapping[tuple[int, int], list[str]]
    ) -> Iterable[TafDatum]:
        for index, line in taf_mapping.items():
            taf = ""
            for i, text in enumerate(line):
                taf += f"{text} "

                yield TafDatum(
                    TX_TN=index,
                    prompt=taf,
                    completion=" ".join(line[i + 1 :]),
                )

    df = (
        pd.DataFrame(generate(taf_series.str.split(r"\s")))  # type: ignore
        .drop_duplicates()
        .astype(str)
    )

    df[["prompt", "completion"]] = (
        df[["prompt", "completion"]]
        .stack()
        # prefix BECMG and TEMPO with the eos & bos tokens
        .str.replace(
            # match whitespace before BECMG or TEMPO
            CHANGE_GROUP_PATTERN,
            # replace with bos + eos
            f"{SpecialTokens.sep_token} ",
            # SpecialTokens.eos_token + SpecialTokens.bos_token,
            regex=True,
        )
        .unstack()
    )
    # add bos to the beginning of the prompt
    df["prompt"] = SpecialTokens.bos_token + df.prompt
    # add eos to the end of the completion
    df["completion"] = df.completion + SpecialTokens.eos_token
    # drop any rows with empty strings
    df = df.drop(df.index[df.completion == ""])
    # write the DataFrame to a jsonl file
    with json_lines_file.open("w") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")


def dataset_dict(
    json_lines_file: StrPath,
    tokenizer: GPT2TokenizerType,
    dataset_dict_path: StrPath,
    batch_size: int,
    text_file: StrPath | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Generate a jsonl file from a text file."""

    def tokenize_function(key: Literal["prompt", "completion"]):
        def wrapper(examples):
            return tokenizer(examples[key], truncation=True, padding=True)

        return wrapper

    if isinstance(json_lines_file, str):
        json_lines_file = Path(json_lines_file)
    if text_file and not json_lines_file.exists():
        # if the jsonl file doesn't exist, generate it from the text file
        _make_json_lines(text_file, json_lines_file)
    elif not json_lines_file.exists():
        # if the jsonl file doesn't exist and a text file
        # was not provided, raise an error
        raise ValueError(
            "Either text_file or json_lines_file must be provided and json_lines_file must exist."
            f"{json_lines_file} does not exist."
        )

    df = pd.read_json(json_lines_file, lines=True)
    train_df = df.sample(frac=1 - test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    train_ds = Dataset.from_pandas(train_df, features=FEATURES, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, features=FEATURES, preserve_index=False)

    (
        DatasetDict(train=train_ds, test=test_ds)
        .map(tokenize_function("prompt"), batch_size=batch_size, batched=True)
        .map(tokenize_function("completion"), batch_size=batch_size, batched=True)
        .save_to_disk(dataset_dict_path)  # type: ignore
    )
