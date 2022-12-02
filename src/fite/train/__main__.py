import json
from typing import Literal

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# project imports
from .. import __version__ as VERSION
from ..pipeline import HyperParameterStrategy, Pipeline
from ..util import CONSTANTS, DEFAULT_DEVICE, ActionStr, ResultRecord, suppress_stdout

# training module imports
from . import fine_tune
from .datasets import Dataset, TextFile
from .filesystem import FileSystem, FileSystemDirectory

# DONT CHANGE
FRAMEWORK = "pt"
BASE_MODEL_NAME = "gpt2"
# TODO: build abstraction for additional tokens and additional special tokens into the file system config
VERBOSE = False


def get_results(
    fs: FileSystem,
    prompts: list[str] | None = None,
    strategies: list[HyperParameterStrategy] | HyperParameterStrategy | None = None,
) -> list[ResultRecord]:

    model, tokenizer = train(fs, push_to_hub=False)

    pipe = Pipeline(
        model=model,
        tokenizer=tokenizer,
        device=DEFAULT_DEVICE,
        max_length=CONSTANTS.MAX_LENGTH,
        num_return_sequences=1,
    )

    if not strategies:
        strategies = list(HyperParameterStrategy)
    elif isinstance(strategies, HyperParameterStrategy):
        strategies = [strategies]

    prompt_examples = fs.config["prompt-examples"]
    if prompts:
        prompt_examples += prompts
    model_name = fs.model_name
    results = []
    for strategy in HyperParameterStrategy:
        if VERBOSE:
            print(f"***** Running {strategy.name} strategy *****")
        generated_text_list = pipe.generate(prompt_examples, strategy=strategy)
        for prompt_text, generated_text in zip(prompt_examples, generated_text_list):
            result = ResultRecord(
                model=model_name,
                prompt_text=prompt_text,
                generated_text=generated_text,
                strategy=strategy.name,
                hyper_parameters=strategy,
            ).evaluate()
            results.append(result)

    return results


def train(
    fs: FileSystem, push_to_hub: bool = False
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:

    if not fs.tokenizer_path.exists():
        if VERBOSE:
            print("***** Tokenizer not found, creating tokenizer... *****")
        fine_tune.tokenizer(fs)

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(fs.tokenizer_path)

    if not fs.model_path.exists():
        if VERBOSE:
            print("***** Model not found, creating model... *****")
        fine_tune.model(fs, tokenizer, push_to_hub=push_to_hub)
    # load the model from the saved path
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(fs.model_path)  # type: ignore
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def batch(fs: FileSystem) -> None:
    # access the prompt examples from the file system config
    prompt_examples = fs.config["prompt-examples"]
    # append the cli input to the prompt examples
    if VERBOSE:
        print(f"***** Prompt Examples: {prompt_examples} *****")

    results = [result.to_dict() for result in get_results(fs, prompt_examples)]

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


def main(fs: FileSystem, push_to_hub=False) -> None:
    """
    numbered bullet points in markdown format

    1. fine-tune the tokenizer
    2. build the dataset_dict
        2.1. build the json lines file
        2.2. if the json lines file doest not exist, attempt to create it from a raw text file
    3. fine-tune the model
    """
    torch.cuda.empty_cache()
    print(f"***** Training model: {fs.model_name} *****")
    print(f"***** From the base model: {fs.base_model} *****")
    print(f"***** Using dataset: {fs.dataset_dict_path} *****")
    # 1.0 -> fine-tune the tokenizer
    if not fs.tokenizer_path.exists():
        print("***** Tokenizer not found, creating tokenizer... *****")
        fine_tune.tokenizer(fs)
    # 1.1 -> load the tokenizer from the filesystem
    tokenizer = fs.get_tokenizer()
    # the dataset will have to be encoded
    # the encode function will be called on the dataset via the map method
    # each call to encode will return a dictionary with kwargs that can be unpacked into
    # the dataset constructor
    encoding_kwargs = {
        "batched": True,
        "batch_size": CONSTANTS.BATCH_SIZE,
        "num_proc": CONSTANTS.NUM_PROC,
    }

    def encode(key: Literal["metadata", "prompt", "completion"]):
        encoding_kwargs.update(
            function=lambda batch: tokenizer(batch[key], truncation=True, padding=True)
        )
        return encoding_kwargs

    # 2.0 -> build the dataset_dict
    if not fs.dataset_dict_path.exists():
        print("***** Dataset not found, creating dataset... *****")
        features = fs.get_features()
        # 2.1 -> build the json lines file
        if not fs.json_lines_file.exists():
            print("***** Json Lines not found, creating json lines... *****")
            if fs.raw_text_file.exists():
                TextFile.from_file(
                    fs.raw_text_file,
                    metadata_pattern=fs.config.get("metadata-pattern", None),
                ).to_jsonl(fs.json_lines_file)
            else:
                raise FileNotFoundError
        # 3.0 -> build the dataset_dict from the json_lines_file
        ds = (
            Dataset.from_json_lines(fs.json_lines_file, features=features)
            .map(**encode("prompt"))
            .map(**encode("completion"))
        )
        if "metadata" in features:
            ds = ds.map(**encode("metadata"))

        ds.save_to_disk(str(fs.dataset_dict_path))
    if not fs.model_path.exists():
        print("***** Model not found, creating model... *****")
        fine_tune.model(fs, tokenizer, push_to_hub=push_to_hub).save_pretrained(
            fs.model_path
        )

    # load the model from the saved path
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(fs.model_path)  # type: ignore
    model.resize_token_embeddings(len(tokenizer))
    # model.to(DEFAULT_DEVICE).resize_token_embeddings(len(tokenizer))
    pipe = Pipeline(
        model=model,
        tokenizer=tokenizer,
        device=DEFAULT_DEVICE,
        max_length=CONSTANTS.MAX_LENGTH,
        num_return_sequences=1,
    )
    result = pipe.generate(
        fs.config["prompt-examples"], strategy=HyperParameterStrategy.GREEDY
    )
    print(result)


if __name__ == "__main__":

    import argparse

    class Namespace(argparse.Namespace):
        def __init__(self, **kwargs):
            for name in kwargs:
                print(name)
                setattr(self, name.replace("-", "_"), kwargs[name])

        file_system: str
        function: str
        version: str
        text: str | list[str]
        dataset_name: str
        push_to_hub: bool
        verbose: bool

    parser = argparse.ArgumentParser(prog="code-predictor")
    parser.add_argument("filesystem", help="The dataset to train")
    parser.add_argument("--dataset-name", type=str, default="gpt2-taf-base1")
    parser.add_argument(
        "--text", type=str, default="TAF KBLV 010600Z 0106/0212 270020G35KT"
    )
    parser.add_argument("--version", type=str, default=VERSION)

    parser.add_argument("--verbose", action=ActionStr.FALSE)
    parser.add_argument("--push-to-hub", action=ActionStr.FALSE)

    args = parser.parse_args(namespace=Namespace())

    # create the file system which includes multiple FileSystems for several models
    fsd = FileSystemDirectory.load_from_pyproject()
    fs = fsd.get(args.filesystem)

    if args.verbose:
        main(fs)
    else:
        with suppress_stdout():
            main(fs)
