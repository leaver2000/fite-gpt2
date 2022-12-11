import argparse
from typing import Literal

import torch

# project imports
from ..pipeline import HyperParameterStrategy, Pipeline
from ..util import CONSTANTS, suppress_stdout, DEFAULT_DEVICE

# training module imports
from . import fine_tune
from .datasets import Dataset, TextFile
from .filesystem import FileSystem, FileSystemDirectory

# DONT CHANGE
FRAMEWORK = "pt"
BASE_MODEL_NAME = "gpt2"
# TODO: build abstraction for additional tokens and additional special tokens into the file system config
VERBOSE = False


class Namespace(argparse.Namespace):
    filesystem: str
    verbose: bool
    force: bool


def train(fs: FileSystem, push_to_hub=False) -> None:
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

    def encode(key: Literal["prompt", "completion"]):
        encoding_kwargs.update(
            function=lambda batch: tokenizer(batch[key], truncation=True, padding=False)
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
        ds.save_to_disk(str(fs.dataset_dict_path))
    if not fs.model_path.exists():
        print("***** Model not found, creating model... *****")
        fine_tune.model(fs, tokenizer, push_to_hub=push_to_hub).save_pretrained(
            fs.model_path
        )


def main(ns: Namespace) -> None:
    fs = FileSystemDirectory.load_from_pyproject("pyproject.toml").get(ns.filesystem)

    if ns.verbose:
        train(fs)
        model = fs.get_model()
        tokenizer = fs.get_tokenizer()
        prompt_examples = fs.config["prompt-examples"]
        pipeline = Pipeline(
            model=model,
            tokenizer=tokenizer,
            device=DEFAULT_DEVICE,
            max_length=CONSTANTS.MAX_LENGTH,
            num_return_sequences=1,
        )
        
        for strategy in [HyperParameterStrategy.GREEDY]:
            print(f"***** {strategy} *****")
            for example in prompt_examples:
                results = pipeline.generate(example, strategy=strategy)
                print(example + "..." + '\n'.join(results)[len(example):], "\n")
            
    else:
        with suppress_stdout():
            train(fs)


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(prog="model-trainer")
    parser.add_argument("filesystem", help="The dataset to train")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    sys.exit(main(parser.parse_args(namespace=Namespace())))
