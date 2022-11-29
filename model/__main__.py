import json

import torch
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    Trainer,
    AddedToken,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets.dataset_dict import DatasetDict
from api.pipeline import CodePredictionPipeline, HyperParameterStrategy


from . import __version__ as VERSION, build, evaluate
from .util import (
    ResultRecord,
    SpecialTokens,
    FileSystem,
    FileSystemDirectory,
    ActionStr,
    CONSTANTS,
    DEFAULT_DEVICE,
)


# DONT CHANGE
FRAMEWORK = "pt"
BASE_MODEL_NAME = "gpt2"
# TODO: build abstraction for additional tokens and additional special tokens into the file system config
VERBOSE = False
ADDITIONAL_TOKENS = ["\u0120TAF", "\u0120BECMG", "\u0120TEMPO"]
# TOKENS to be ignored by the tokenizer
ADDITIONAL_SPECIAL_TOKENS = SpecialTokens.include(
    AddedToken("LAST", single_word=True, lstrip=True, rstrip=True),
    AddedToken("NO", single_word=True, lstrip=True, rstrip=True),
    AddedToken("AMDS", single_word=True, lstrip=True, rstrip=True),
    AddedToken("RMK", single_word=True, lstrip=True, rstrip=True),
)


def fine_tune_tokenizer(fs: FileSystem) -> None:
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        do_basic_tokenize=False,
    )

    tokenizer.add_tokens(ADDITIONAL_TOKENS)  # type: ignore
    tokenizer.add_special_tokens(ADDITIONAL_SPECIAL_TOKENS)  # type: ignore
    tokenizer.save_pretrained(fs.tokenizer)


def fine_tune_model(
    fs: FileSystem,
    tokenizer: GPT2TokenizerFast,
    push_to_hub: bool = False,
) -> None:
    torch.cuda.empty_cache()
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
        BASE_MODEL_NAME,
        # GPT2Config -> https://huggingface.co/transformers/v3.5.1/model_doc/gpt2.html#gpt2config
        config=GPT2Config(
            activation_function="gelu_new",
            layer_norm_eps=1e-05,
        ),
    ).to(  # type: ignore
        DEFAULT_DEVICE
    )
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    # if dataset does not exist, create it
    if not fs.dataset_dict.exists():
        print("***** Dataset not found, creating dataset... *****")
        if not fs.json_lines.exists():
            print("***** json lines file not found, creating json lines... *****")
            build.json_lines(  # TODO: add a config file to the file map for additional tokens n stuff
                lambda s: (
                    s.str.extract(build.TEMPERATURE_GROUP_PATTERN, expand=True)
                    .stack()
                    .str.replace("M", "-")
                    .unstack()
                ),
                text_file=fs.raw_text,
                jsonl_file=fs.json_lines,
            )
    build.dataset_dict(fs, tokenizer)

    print("***** Loading dataset... *****")
    tokenized_ds = DatasetDict.load_from_disk(str(fs.dataset_dict))
    # ###  Trainer Setup ###
    # configure the trainer arguments
    training_arguments = TrainingArguments(
        run_name=fs.name,
        output_dir=str(fs.model),
        overwrite_output_dir=True,
        per_device_train_batch_size=CONSTANTS.BATCH_SIZE,
        per_device_eval_batch_size=CONSTANTS.BATCH_SIZE,
        # begin_suppress_tokens =tokenizer.all_special_ids,
        #
        num_train_epochs=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=0,
    )
    # configure the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        # mlm_probability=0.15,
        pad_to_multiple_of=20,
        return_tensors=FRAMEWORK,
    )
    model.train()
    # create trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=tokenized_ds["train"],  # type: ignore
        eval_dataset=tokenized_ds["test"],  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=None,  # type:ignore ((EvalPrediction) -> Dict[Unknown, Unknown]) | None = None,
        model_init=None,  # type:ignore () -> PreTrainedModel = None,
        callbacks=None,  # type:ignore List[TrainerCallback] | None = None,
        optimizers=(
            None,
            None,
        ),  # type:ignore Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics=None,  # type:ignore (Tensor, Tensor) -> Tensor = None
    )
    # train the model
    trainer.train()
    # save model
    model.save_pretrained(fs.model)
    if push_to_hub:
        model.push_to_hub(fs.name)
        repo_url = trainer.push_to_hub()
        print(f"Pushed model to 🤗 {repo_url}")


def get_results(
    fs: FileSystem,
    prompts: list[str],
    strategies: list[HyperParameterStrategy] | None = None,
) -> list[ResultRecord]:

    model, tokenizer = train(fs, push_to_hub=False)

    pipe = CodePredictionPipeline(
        model=model,
        tokenizer=tokenizer,
        device=DEFAULT_DEVICE,
        max_length=CONSTANTS.MAX_LENGTH,
        num_return_sequences=1,
    )
    if strategies is None:
        strategies = list(HyperParameterStrategy)

    results = []
    for strategy in [HyperParameterStrategy.GREEDY]:
        if VERBOSE:
            print(f"***** Running {strategy.name} strategy *****")

        generated_text_list = pipe.generate_forecast(prompts, strategy=strategy)
        for prompt_text, generated_text in zip(prompts, generated_text_list):
            result = ResultRecord(
                model=fs.model_name,
                prompt_text=prompt_text,
                generated_text=generated_text,
                score=evaluate.taf(prompt_text, generated_text),  # TODO: ...
                strategy=strategy.name,
                hyper_parameters=strategy,
            )
            results.append(result)

    return results


def train(
    fs: FileSystem, push_to_hub: bool = False
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:

    if not fs.tokenizer.exists():
        print("***** Tokenizer not found, creating tokenizer... *****")
        fine_tune_tokenizer(fs)

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(fs.tokenizer)

    if not fs.model.exists():
        print("***** Model not found, creating model... *****")
        fine_tune_model(fs, tokenizer, push_to_hub=False)
    # load the model from the saved path
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(fs.model)  # type: ignore
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def batch(fs: FileSystem) -> None:
    # access the prompt examples from the file system config
    prompt_examples = fs.config["prompt_examples"]
    # append the cli input to the prompt examples
    print(f"***** Prompt Examples: {prompt_examples} *****")

    results = [result.to_dict() for result in get_results(fs, prompt_examples)]

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


def main(fs: FileSystem, text_input: str | list[str]) -> None:
    if isinstance(text_input, str):
        text_input = [text_input]
    results = get_results(fs, text_input)
    for result in results:
        print("\n ".join(result.generated_text) + "\n")


if __name__ == "__main__":

    import argparse

    class Namespace(argparse.Namespace):
        function: str
        version: str
        text: str | list[str]
        dataset_name: str
        push_to_hub: bool
        verbose: bool

    parser = argparse.ArgumentParser(prog="code-predictor")
    parser.add_argument("function", help="run functions")
    parser.add_argument("--dataset-name", type=str, default="taf")
    parser.add_argument(
        "--text", type=str, default="TAF KBLV 010600Z 0106/0212 270020G35KT"
    )
    parser.add_argument("--version", type=str, default=VERSION)

    parser.add_argument("--verbose", action=ActionStr.FALSE)
    parser.add_argument("--push-to-hub", action=ActionStr.FALSE)

    args = parser.parse_args(namespace=Namespace())

    if args.verbose:
        VERBOSE = True
    # create the file system which includes multiple FileSystems for several models
    fsd = FileSystemDirectory(BASE_MODEL_NAME, args.version)
    # select the model to use based on the dataset name
    fs = fsd.get(args.dataset_name)

    match args.function:
        case "main":
            main(fs, args.text)
        case "train":
            train(fs, push_to_hub=args.push_to_hub)
        case "batch":
            batch(fs)
        case _:
            raise ValueError(f"Function {args.function} not found")
