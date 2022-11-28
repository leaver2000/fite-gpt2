import os
import re
import json
from typing import Iterable, NamedTuple

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

from . import __version__ as version, build, evaluate
from .util import (
    SpecialTokens,
    get_file_system,
    DEFAULT_DEVICE,
    MAX_LENGTH,
    BATCH_SIZE,
)

from api.pipeline import CodePredictionPipeline, HyperParameterStrategy, HyperParameters


# DONT CHANGE
FRAMEWORK = "pt"
BASE_MODEL_NAME = "gpt2"
# the dataset name should be mapped to a jsonl file in the store/data
DATASET_NAME = os.getenv("FITE_DATASET_NAME", "taf")
FILE_SYSTEM = get_file_system(BASE_MODEL_NAME, version)
FILE_MAP = FILE_SYSTEM.get(DATASET_NAME)
if not FILE_MAP:
    raise OSError(
        f"Dataset name {DATASET_NAME} not found in FILE_MAP. Please add it to the map.\n"
        f"store/data/{DATASET_NAME}/-training-data.txt is required for this script to run that DATASET"
    )

MODEL_NAME, RAW_TEXT, JSONL_FILE, DATASET_PATH, TOKENIZER_PATH, MODEL_PATH = FILE_MAP
# TODO: add a config file to the file map for additional tokens n stuff

# TYPES
ADDITIONAL_TOKENS = ["\u0120TAF", "\u0120BECMG", "\u0120TEMPO"]
# TOKENS to be ignored by the tokenizer
ADDITIONAL_SPECIAL_TOKENS = SpecialTokens.include(
    AddedToken("LAST", single_word=True, lstrip=True, rstrip=True),
    AddedToken("NO", single_word=True, lstrip=True, rstrip=True),
    AddedToken("AMDS", single_word=True, lstrip=True, rstrip=True),
    AddedToken("RMK", single_word=True, lstrip=True, rstrip=True),
)


def fine_tune_tokenizer() -> None:
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        do_basic_tokenize=False,
    )

    tokenizer.add_tokens(ADDITIONAL_TOKENS)  # type: ignore
    tokenizer.add_special_tokens(ADDITIONAL_SPECIAL_TOKENS)  # type: ignore
    tokenizer.save_pretrained(TOKENIZER_PATH)


def fine_tune_model(tokenizer: GPT2TokenizerFast) -> None:
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
    if not DATASET_PATH.exists():
        print("***** Dataset not found, creating dataset... *****")
        if not JSONL_FILE.exists():
            print("***** json lines file not found, creating json lines... *****")
            build.json_lines(  # TODO: add a config file to the file map for additional tokens n stuff
                lambda s: s.str.extract(build.TEMPERATURE_GROUP_PATTERN, expand=True)
                .stack()
                .str.replace("M", "-")
                .unstack(),
                "\n\n###\n\n",
                sep_pattern="\n",
                text_file=RAW_TEXT,
                jsonl_file=JSONL_FILE,
            )

        build.dataset_dict(
            JSONL_FILE,
            tokenizer,
            DATASET_PATH,
            BATCH_SIZE,
        )

    print("***** Loading dataset... *****")
    tokenized_ds = DatasetDict.load_from_disk(DATASET_PATH)  # type: ignore
    # ###  Trainer Setup ###
    # configure the trainer arguments
    training_arguments = TrainingArguments(
        run_name=MODEL_NAME,
        output_dir=MODEL_PATH,  # type: ignore
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
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
    model.save_pretrained(MODEL_PATH)  # type: ignore
    model.push_to_hub(MODEL_NAME)  # type: ignore


def handle_prediction(
    result: Iterable[dict[str, str]] | list[dict[str, str]]
) -> Iterable[list[str]]:
    for item in result:
        if isinstance(item, list):
            yield from handle_prediction(item)
        else:
            yield re.split(r"(?=BECMG|TEMPO)", item["generated_text"])


class ResultOutput(NamedTuple):
    """ResultOutput is a named tuple that contains the output of the model."""

    model: str
    prompt_text: str
    generated_text: list[str]
    score: float
    strategy: str
    hyper_parameters: HyperParameterStrategy | HyperParameters


def train() -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    if not TOKENIZER_PATH.exists():
        print("***** Tokenizer not found, creating tokenizer... *****")
        fine_tune_tokenizer()

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        TOKENIZER_PATH,
    )

    if not MODEL_PATH.exists():
        print("***** Model not found, creating model... *****")
        fine_tune_model(tokenizer)
    # load the model from the saved path
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(MODEL_PATH)  # type: ignore
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def get_results(prompts: list[str]) -> list[ResultOutput]:

    model, tokenizer = train()

    pipe = CodePredictionPipeline(
        model=model,
        tokenizer=tokenizer,
        device=DEFAULT_DEVICE,
        max_length=MAX_LENGTH,
        num_return_sequences=1,
    )

    def generate_forecast_output(*strategies: HyperParameterStrategy):

        for strategy in strategies:
            generated_text_list = pipe.generate_forecast(prompts, strategy=strategy)
            for prompt_text, generated_text in zip(prompts, generated_text_list):
                yield ResultOutput(
                    model=MODEL_NAME,
                    prompt_text=prompt_text,
                    generated_text=generated_text,
                    score=evaluate.taf(prompt_text, generated_text),  # TODO: ...
                    strategy=strategy.name,
                    hyper_parameters=strategy,
                )

    return list(generate_forecast_output(*HyperParameterStrategy))


def batch(text_input: str) -> None:
    text_prompts = [
        (
            "TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS\n"
            "TEMPO 0200/0206 5000"
        ),
        ("TAF KGTB 251700Z 2517/2623 26012G20KT"),
        (
            "TAF KGTB 251700Z 2517/2623 26012G20KT 9999 OVC008 QNH2970INS\n"
            "BECMG 2519/2520 27009KT 9999 SCT009 OVC015 QNH2976INS\n"
            "BECMG 2610/2611 VRB06KT 9999 BKN015 OVC025 QNH2991INS"
        ),
        ("TAF KMTC 252000Z 2520/2702 29012G20KT 9999 SKC"),
        (
            "TAF PASY 251400Z 2514/2620 11006KT 9999 FEW030 FEW045 SCT100 QNH3002INS\n"
            "BECMG 2519/2520"
        ),
        text_input,
    ]

    results = [result._asdict() for result in get_results(text_prompts)]

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


def validate():
    with open("results.json") as f:
        results = (ResultOutput(**result) for result in json.load(f))

    for result in results:
        score = 0.0
        # looking at the most last line in the prompt text
        prompt = result.prompt_text.split("\n")[-1]
        if "TS" in prompt:
            # only one line should start with that prompt
            (last_prompt_line,) = (
                line for line in result.generated_text if line.startswith(prompt)
            )
            # if the prompt has a TS in it then the generated text should have a CB remark
            if "CB" in last_prompt_line:
                score += 1
            else:
                score -= 1
            # there should not be any lower case letters in the generated text
            if all(line.isupper() for line in result.generated_text):
                score += 1
            else:
                score -= 1
        yield result._replace(score=score)


def main(text_input: str) -> None:
    results = get_results([text_input])
    for result in results:
        print("\n ".join(result.generated_text) + "\n")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # positional argument for switch
    parser.add_argument("function", default="main")
    # function arguments
    parser.add_argument(
        "--text", type=str, default="TAF KBLV 010600Z 0106/0212 270020G35KT"
    )
    args = parser.parse_args()
    match args.function:
        case "main":
            main(args.text)
        case "train":
            train()
        case "batch":
            batch(args.text)
        case "validate":
            import pandas as pd

            df = (
                pd.DataFrame(validate())
                .drop(columns=["hyper_parameters", "model"])
                .set_index("strategy")
            )
            for name, prompt_text, generated_text, score in df.sort_values(
                by=["score"], ascending=True
            ).itertuples():
                generated_text = "\n ".join(generated_text)
                print(
                    f"""
{name=} {prompt_text=} {score=}
{generated_text}"""
                )

        case _:
            raise ValueError(f"Function {args.function} not found")
