import json
from pathlib import Path
import torch

# project imports
from ..pipeline import HyperParameterStrategy, Pipeline
from ..util import CONSTANTS, DEFAULT_DEVICE
from .filesystem import FileSystem, FileSystemDirectory


def create_output_results(fs: FileSystem, strategy:HyperParameterStrategy, out_dir:str|None=None):
    """
    create the output results
    """
    torch.cuda.empty_cache()
    # load the model from the saved path
    model = fs.get_model()
    tokenizer = fs.get_tokenizer()
    prompt_examples = fs.config["prompt-examples"]
    pipe = Pipeline(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        max_length=CONSTANTS.MAX_LENGTH,
        num_return_sequences=1,
    )


    pipe = Pipeline(
        model=model,
        tokenizer=tokenizer,
        device=DEFAULT_DEVICE,
        max_length=CONSTANTS.MAX_LENGTH,
        num_return_sequences=1,
    )


    results = []
    for prompt in prompt_examples:
        completion = pipe.generate(prompt, strategy=strategy)
        results.append({
            "prompt": prompt,
            "completion": "\n".join(completion)[len(prompt) :],
            "strategy": strategy.name,
        })
    return results


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filesystem", help="The dataset to train")
    # add the strategies as arguments
    for strategy in HyperParameterStrategy:
        parser.add_argument(
            f"--{strategy.name.lower().replace('_', '-')}",
            action="store_true",
            help=f"{strategy=}",
        )
    # an option to use all strategies
    parser.add_argument("--all", action="store_true", help="Use all strategies")
    # the output directory
    parser.add_argument("--out-dir", help="The output directory", default=None)
    # load the filesystem directory
    fsd = FileSystemDirectory.load_from_pyproject()
    # arguments to a dictionary
    args = vars(parser.parse_args())
    # get the filesystem
    fs = fsd.get(args.pop("filesystem"))
    # pop the arguments
    do_all = args.pop("all")
    out_dir = args.pop("out_dir", None)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    strategies = [HyperParameterStrategy[k.upper()] for k, v in args.items() if v or do_all]
    
    for strategy in strategies:
        print(f"*****running {strategy} *****")
        results = create_output_results(fs, strategy)
        if out_dir:
            out_file = out_dir / f"{strategy.name.lower()}.json"
            with out_file.open( "w") as f:
                for result in results:
                    f.write(json.dumps(result))
                    f.write("\n")


if __name__ == "__main__":
    main()
