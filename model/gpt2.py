"""



"""
import re
import json
from typing import Iterable, NamedTuple
from pathlib import Path


import torch
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AddedToken

from datasets.dataset_dict import DatasetDict

from ._dataset_builder import get_dataset_dict
from ._pipeline import CodePredictionPipeline, HyperParameterStrategy, HyperParameters

from .util import (
    unpack_paths,
    SpecialTokens,
)


# DONT CHANGE
FRAMEWORK = "pt"
PRE_TRAINED_MODEL_NAME = "gpt2"
# VERSIONING
DATASET_NAME = "taf"
MODEL_VERSION = 1
VERSION = f"{MODEL_VERSION}.0"
MODEL_PATH, DATASET_PATH, RAW_TEXT, JSON_LINES_FILE = unpack_paths(
    DATASET_NAME, VERSION
)
MODEL_NAME = f"{PRE_TRAINED_MODEL_NAME}-{DATASET_NAME}-{VERSION}"

# RUNTIME VARIABLES
MAX_LENGTH = 256
BATCH_SIZE = 8

# TYPES
StrPath = str | Path

ADDITIONAL_TOKENS = ("\u0120TAF", "\u0120BECMG", "\u0120TEMPO")
# TOKENS to be ignored by the tokenizer
ADDITIONAL_SPECIAL_TOKENS = SpecialTokens.include(
    AddedToken("LAST", single_word=True, lstrip=True, rstrip=True),
    AddedToken("NO", single_word=True, lstrip=True, rstrip=True),
    AddedToken("AMDS", single_word=True, lstrip=True, rstrip=True),
    AddedToken("RMK", single_word=True, lstrip=True, rstrip=True),
)


# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)


def get_language_model(
    base_model: StrPath = PRE_TRAINED_MODEL_NAME,
    *args,
    config: GPT2Config | None = None,
    verbose: bool = False,
    **kwargs,
) -> GPT2LMHeadModel:

    model = GPT2LMHeadModel.from_pretrained(
        base_model,
        *args,
        config=config,
        **kwargs,
    )
    # should always be in eval mode
    assert isinstance(model, GPT2LMHeadModel)
    if verbose:
        print(model.config)
    return model.cuda(device)


def fine_tune_model(tokenizer: GPT2Tokenizer) -> None:
    torch.cuda.empty_cache()
    model = get_language_model(
        PRE_TRAINED_MODEL_NAME,
        # GPT2Config -> https://huggingface.co/transformers/v3.5.1/model_doc/gpt2.html#gpt2config
        config=GPT2Config(
            activation_function="gelu_new",
            layer_norm_eps=1e-05,
        ),
    )
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    # if dataset does not exist, create it
    if not DATASET_PATH.exists():
        print("***** Dataset not found, creating dataset... *****")
        (
            get_dataset_dict(RAW_TEXT, JSON_LINES_FILE)
            .map(
                lambda x: tokenizer(x["prompt"], truncation=True, padding=True),
                batch_size=BATCH_SIZE,
                batched=True,
            )
            .map(
                lambda x: tokenizer(x["completion"], truncation=True, padding=True),
                batch_size=BATCH_SIZE,
                batched=True,
            )
            .save_to_disk(DATASET_PATH)  # type: ignore
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


def run(text_input: str) -> list[ResultOutput]:
    """
    ### example:
    provided the string below the model correctly encoded the temporary group
    by including a visibility obstruction and a lower ceiling.
    this is a very common use case when encoding TEMPO groups during showers

    The third line in the taf has a-lot more randomness to it.
    """
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=2,
        do_basic_tokenize=False,
    )

    tokenizer.add_tokens(ADDITIONAL_TOKENS)  # type: ignore
    tokenizer.add_special_tokens(ADDITIONAL_SPECIAL_TOKENS)  # type: ignore
    if not MODEL_PATH.exists():
        # if a new model is being trained fine tune the model
        fine_tune_model(tokenizer)
    # load the model from the saved path
    model = get_language_model(MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer))

    print("***** batch_decode -> ... *****")
    pipe = CodePredictionPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=MAX_LENGTH,
        num_return_sequences=3,
    )

    def generate_forecast_output(*strategies: HyperParameterStrategy):
        for strategy in strategies:
            for forecast in pipe.generate_forecast(text_input, strategy=strategy):
                yield ResultOutput(
                    model=MODEL_NAME,
                    prompt_text=text_input,
                    generated_text=forecast,
                    score=0.0,  # TODO: ...
                    strategy=strategy.name,
                    hyper_parameters=strategy,
                )

    return list(generate_forecast_output(*HyperParameterStrategy))


def pipeline() -> CodePredictionPipeline:
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=2,
        do_basic_tokenize=False,
    )

    tokenizer.add_tokens(ADDITIONAL_TOKENS)  # type: ignore
    tokenizer.add_special_tokens(ADDITIONAL_SPECIAL_TOKENS)  # type: ignore
    model = get_language_model(MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer))
    return CodePredictionPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=MAX_LENGTH,
        num_return_sequences=3,
    )


def many(text_input: str):
    text_promps = (
        "TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS\nTEMPO 0200/0206 5000",
        "TAF KGTB 251700Z 2517/2623 26012G20KT",
        (
            "TAF KGTB 251700Z 2517/2623 26012G20KT 9999 OVC008 QNH2970INS\n"
            "BECMG 2519/2520 27009KT 9999 SCT009 OVC015 QNH2976INS\n"
            "BECMG 2610/2611 VRB06KT 9999 BKN015 OVC025 QNH2991INS"
        ),
        "TAF KMTC 252000Z 2520/2702 29012G20KT 9999 SKC",
        "TAF PASY 251400Z 2514/2620 11006KT 9999 FEW030 FEW045 SCT100 QNH3002INS\nBECMG 2519/2520",
        text_input,
    )
    results = []
    for text_input in text_promps:
        results.extend([result._asdict() for result in run(text_input)])

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
    run(text_input)


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
            result = main(args.text)
            print(result)
        case "many":
            many(args.text)
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
