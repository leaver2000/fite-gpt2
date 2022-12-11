import json
import argparse
from pathlib import Path
import dataclasses
import torch

# project imports
from .._argparse import ArgumentParser, DefaultNamespace, argument
from ..pipeline import HyperParameterStrategy, Pipeline
from ..util import CONSTANTS, DEFAULT_DEVICE

# training module imports
from .filesystem import FileSystem, FileSystemDirectory

__all__ = ["get_results"]


def get_results(fs: FileSystem, strategy: HyperParameterStrategy):
    """
    create the output results
    """
    torch.cuda.empty_cache()
    # load the model from the saved path
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

    results = []
    for prompt in prompt_examples:
        completion = pipeline.generate(prompt, strategy=strategy)
        results.append(
            {
                "prompt": prompt,
                "completion": "\n".join(completion)[len(prompt) :],
                "strategy": strategy.name,
            }
        )
    return results


@dataclasses.dataclass
class Namespace(DefaultNamespace):
    filesystem: str = argument(help="The dataset to train")
    all: bool = argument(False, action="store_true")
    save_results: bool = argument(False, action="store_true", help="Save the results")


def main(ns: Namespace) -> None:
    # load the model filesystem
    fs = FileSystemDirectory.load_from_pyproject("pyproject.toml").get(ns.filesystem)

    out_dir = None
    if ns.save_results:
        out_dir = Path.cwd() / "results" / ns.filesystem
        out_dir.mkdir(parents=True, exist_ok=True)
    #
    if ns.all:
        strategies = HyperParameterStrategy
    else:
        strategies = [
            strategy
            for strategy in HyperParameterStrategy
            if getattr(ns, strategy.name.lower(), False)
        ]
    for strategy in strategies:
        print(f"***** {strategy} *****")
        results = get_results(fs, strategy)
        # print(f"***** {strategy} *****")
        if ns.verbose:
            print(results)
        if out_dir:
            out_file = out_dir / f"{strategy.name.lower()}.jsonl"
            with out_file.open("w") as f:
                for result in results:
                    f.write(json.dumps(result))
                    f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(Namespace())
    #  add the strategies to the parser
    for strategy in HyperParameterStrategy:
        parser.add_argument(f"--{strategy.name.lower()}", action="store_true")

    raise SystemExit(
        # parse the arguments and add the strategies to the namespace
        main(parser.parse_args())
    )
