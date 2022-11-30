import json
from pathlib import Path
from typing import TypedDict, Iterable, Literal, Callable, Any

import pandas as pd
import datasets
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers import GPT2TokenizerFast

from .util import SpecialTokens, CONSTANTS, FileSystem

__all__ = ["dataset_dict", "json_lines"]

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


def _generate_datums(rows: list[dict[str, str | dict[str, str]]]) -> Iterable[Datum]:
    """
    iterate over the rows splitting each word, the split text is used to create the prompt and completion.
    the __text__ is popped from each dict to create the prompt and completion.
    the remaining dict is used as the metadata.
    """
    for row in rows:
        prompt = ""
        text_list = row.pop("__text__").split()  # type: ignore
        # row = toml.dumps(row)
        for i, text in enumerate(text_list):
            prompt += f"{text} "

            yield Datum(
                metadata=row,
                prompt=prompt,
                completion=" ".join(text_list[i + 1 :]),
            )


MetadataCallback = Callable[[pd.Series], pd.Series | pd.DataFrame]


def json_lines(
    metadata_func: MetadataCallback,
    *,
    split_pattern: str = SPLIT_PATTERN,
    sep_pattern: str | None = CHANGE_GROUP_PATTERN,
    text_file: Path,
    jsonl_file: Path,
) -> None:
    """Generate a jsonl file from a text file.
    >>> build.json_lines(
    ...     lambda s:(
    ...     s.str.extract(r"\\sTX?(?P<max_temp>M?\\d{2})\\/\\d{4}Z\\sTN?(?P<min_temp>M?\\d{2})\\/\\d{4}Z$")
    ...     .stack().str.replace("M", "-").unstack()
    ...     ),
    ...     text_file="data/2021-01-01.txt",
    ...     jsonl_file="data/2021-01-01.jsonl"
    ...     )

    Parameters
    ----------
    index_callback : Callable[[pd.Series[str]],pd.Series[str] | pd.DataFrame]
        A function that takes a series of strings and returns a series of strings or a dataframe.
        The dataframe should have the same number of rows as the series.
        The dataframe will be used to create a multi-index.
    split_pattern : str, optional
        A regex pattern to split the text file into separate entries, by default SPLIT_PATTERN
    sep_pattern : str | None, optional
        A regex pattern to split each entry into a prompt and completion, by default None
    **kwargs : Unpack[PathMap]
        A mapping of text_file to jsonl_file.
        The text_file will be read and the jsonl_file will be created.
    """
    # jsonl_file, text_file = (Path(kwargs[key]) for key in ("jsonl_file", "text_file"))

    with text_file.open("r") as f:
        s = pd.Series(f.read().split(split_pattern), name="__text__").str.strip()
    df = pd.DataFrame(
        _generate_datums(s.to_frame().join(metadata_func(s)).to_dict(orient="records"))  # type: ignore
    )
    # add bos to the beginning of the prompt and eos to the end of the completion
    df["prompt"] = SpecialTokens.bos_token + df.prompt
    df["completion"] = df.completion + SpecialTokens.eos_token
    # drop any duplicates in the prompt and completion
    prompt_completion = df[["prompt", "completion"]].drop_duplicates()
    if sep_pattern:
        # if there is a sep_pattern, replace the sep_pattern with the sep_token
        prompt_completion = (
            prompt_completion.stack()
            .str.replace(sep_pattern, SpecialTokens.sep_token, regex=True)
            .str.strip()
            .unstack()
        )
    # drop unmodified prompts and completions and drop duplicates from the metadata

    df = (
        df.drop(columns=["prompt", "completion"])
        .loc[prompt_completion.index, :]
        .join(prompt_completion)
    ).astype(str)

    with jsonl_file.open("w") as f:
        for line in df.to_dict(orient="records"):
            f.write(json.dumps(line) + "\n")


# class CallbackEngine(TypedDict):
#     TAF: MetadataCallback


# Engine = CallbackEngine(
#     TAF=lambda s: s.str.extract(TEMPERATURE_GROUP_PATTERN, expand=True)
#     .stack()
#     .str.replace("M", "-")
#     .unstack()
# )


def dataset_dict(
    fs: FileSystem,
    tokenizer: GPT2TokenizerFast,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Generate a jsonl file from a text file."""

    def tokenize_function(key: Literal["prompt", "completion"]):
        def wrapper(examples):
            return tokenizer(examples[key], truncation=True, padding=True)

        return wrapper

    print("*** tokenizing ***")
    if not fs.json_lines.exists():
        # if the jsonl and text are not in the filesystem, raise an error
        raise ValueError(
            "Either text_file or json_lines_file must be provided and json_lines_file must exist."
            f"{fs.json_lines} does not exist."
        )

    df = pd.read_json(fs.json_lines, lines=True)
    train_df = df.sample(frac=1 - test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    train_ds = Dataset.from_pandas(train_df, features=FEATURES, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, features=FEATURES, preserve_index=False)

    (
        DatasetDict(train=train_ds, test=test_ds)
        .map(tokenize_function("prompt"), batch_size=CONSTANTS.BATCH_SIZE, batched=True)
        .map(
            tokenize_function("completion"),
            batch_size=CONSTANTS.BATCH_SIZE,
            batched=True,
        )
        .save_to_disk(str(fs.dataset_dict))
    )
