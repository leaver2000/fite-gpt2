"""



"""
import numpy as np
import pandas as pd
import torch
from typing import Iterable
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from transformers import Trainer, TrainingArguments, AddedToken
from transformers import DataCollatorForLanguageModeling

from datasets.dataset_dict import DatasetDict
from typing import NamedTuple
import re
from typing import Iterable
from ._dataset_builder import get_dataset_dict
from ._pipeline import CodePredictionPipeline
from .util import (
    unpack_paths,
    SpecialTokens,
)
from pathlib import Path

StrPath = str | Path
# DONT CHANGE
FRAMEWORK = "pt"
PRE_TRAINED_MODEL_NAME = "gpt2"
# VERSIONING
MODEL_VERSION = 1
TOKENIZER_VERSION = 5
DATASET_VERSION = 12
VERSION = f"0.0.3dev-{MODEL_VERSION}.{TOKENIZER_VERSION}.{DATASET_VERSION}"
MODEL_NAME = f"{PRE_TRAINED_MODEL_NAME}-taf-{VERSION}"
MODEL_PATH, DATASET_PATH, JSON_LINES_FILE = unpack_paths(VERSION)
# RUNTIME VARIABLES
MAX_LENGTH = 256
BATCH_SIZE = 8


ADDITIONAL_TOKENS = ("\u0120TAF", "\u0120BECMG", "\u0120TEMPO")
# TOKENS to be ignored by the tokenizer
ADDITIONAL_SPECIAL_TOKENS = SpecialTokens.include(
    AddedToken("LAST", single_word=True, lstrip=True, rstrip=True),
    AddedToken("NO", single_word=True, lstrip=True, rstrip=True),
    AddedToken("AMDS", single_word=True, lstrip=True, rstrip=True),
    AddedToken("RMK", single_word=True, lstrip=True, rstrip=True),
)
# special tokens


class HyperParameters(NamedTuple):
    """
    HyperParameters
    """

    repetition_penalty: float
    "repetition penalty is a hyperparameter that controls the model's repetition of the same token. The higher the value, the more repetitive the text will be."
    temperature: float
    "temperature is a hyperparameter that controls the randomness of the model's predictions. The higher the value, the more random the text will be."
    top_k: int
    "top_k is a hyperparameter that controls the number of tokens that the model will consider when predicting the next token. The higher the value, the more random the text will be."
    top_p: float
    "top_p is a hyperparameter that controls the number of tokens that the model will consider when predicting the next token. The higher the value, the more random the text will be."


HYPER_PARAMETERS = HyperParameters(
    repetition_penalty=100.0, temperature=0.1, top_k=1, top_p=0.2
)
REPETITION_PENALTY, TEMPERATURE, TOP_K, TOP_P = HYPER_PARAMETERS
# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
# load tokenizer


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
        **kwargs,
        config=config,
    )
    # should always be in eval mode
    assert isinstance(model, GPT2LMHeadModel)
    if verbose:
        print(model.config)
    return model.cuda(device)


def fine_tune_model(tokenizer: GPT2Tokenizer) -> None:
    torch.cuda.empty_cache()
    model = get_language_model(
        # GPT2Config -> https://huggingface.co/transformers/v3.5.1/model_doc/gpt2.html#gpt2config
        config=GPT2Config(
            activation_function="gelu_new",
            layer_norm_eps=1e-05,
        )
    )
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    # if dataset does not exist, create it
    if not DATASET_PATH.exists():
        print("***** Dataset not found, creating dataset... *****")
        (
            get_dataset_dict(JSON_LINES_FILE)
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
        model_init=None,  # type:ignore () -> PreTrainedModel = None,
        compute_metrics=None,  # type:ignore ((EvalPrediction) -> Dict[Unknown, Unknown]) | None = None,
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


def handle_prediction(result) -> Iterable[list[str]]:
    for item in result:
        if isinstance(item, list):
            yield from handle_prediction(item)
        else:
            yield re.split(r"(?=BECMG|TEMPO)", item["generated_text"])


def main(text: str) -> None:
    """
    ### example:
    provided the string below the model correctly encoded the temporary group by including a visibility obstruction and a lower ceiling.
    this is a very common use case when encoding TEMPO groups during showers

    The third line in the taf has a-lot more randomness to it.

    >>> python -m model.gpt2 --text "TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS\\nTEMPO 0200/0206 5000"
    [[{'generated_text': 'TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS\\nTEMPO 0200/0206 5000 BR BKN015\\nBECMG 0314 VRG18650 510004 510013 650726 521044 510353 SN SCT024 620303 530154 540403 FEW017 VCSH 0512Z 54 01006W 4800 RA SKC WS009CB 56012QLD035 520204 FG 9000 DVRS 621958 610002 623504 3'}]]
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

    encoding = tokenizer(
        text,
        return_tensors=FRAMEWORK,
        truncation=True,
    ).to(device)
    # model.pad_token_id = tokenizer.eos_token_id
    outputs = model.generate(encoding.input_ids, do_sample=False, max_length=MAX_LENGTH)
    print("***** batch_decode -> ... *****")
    for res in (
        "\n ".join(re.split(r"(?=BECMG|TEMPO)", item))
        for item in tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ):
        print(res)
    # print(prediction)
    pipe = CodePredictionPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        temperature=TEMPERATURE,
        max_length=MAX_LENGTH,
        num_return_sequences=3,
    )
    print("***** pipe.forecast -> ... *****")
    for prediction in handle_prediction(pipe.forecast(text)):
        print("\n ".join(prediction))
        print()
    print("***** pipe.__call__ -> ... *****")
    for prediction in handle_prediction(pipe(text)):
        print("\n ".join(prediction))
        print()
    # print(list(handle_prediction(prediction)))
    # print(pipe(
    # text,
    #     # temperature=.9,
    #     # return_full_text=True, #  If set to False only added text is returned, otherwise the full text is returned Only meaningful if return_text is set to True.
    #     # # num_return_sequences=3,
    #     # early_stopping=False,
    #     # clean_up_tokenization_spaces = False,
    #     # handle_long_generation = "hole",
    #     # handle_
    #     ))
    # print(tokenizer.get_vocab().keys())
    # with open("t.txt", "w") as f:
    #     for key in tokenizer.get_vocab().keys():
    #         f.write(key + "\n\n")
    #     # f.write(tokenizer.get_vocab().keys())
    # print(prediction)


def test_tokenizer(
    text: str, additional_special_tokens: list[AddedToken] = ADDITIONAL_SPECIAL_TOKENS
) -> None:
    """
    test the tokenizer
    """
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=2,
        # TODO: adding a vocab file to the tokenizer
        # vocab_files={"vocab_file": "vocab.txt"},
        # vocab_file=os.path.join(MODEL_PATH, "vocab.json"),
    )
    special_tokens = SpecialTokens.to_dict()
    special_tokens["additional_special_tokens"] = additional_special_tokens  # type: ignore
    tokenizer.add_special_tokens(special_tokens)
    print(tokenizer.tokenize(text, truncation=True, padding=True))
    tokens = (
        "TAF",
        "\nBECMG",
        "BECMG",
        "\nTEMPO",
        "TEMPO",
        "VRB",
        "VRB06KT",
        "18010G15KT" "BKN030",
        "BKN030 ",
    )

    for token in tokens:
        print(token, tokenizer.tokenize(token), sep="=")
    with open("store/training-data-v2.txt", "r") as f:
        lines = f.read().split("\n\n###\n\n")
        # print(lines)
        for line in lines:
            lines = line.strip()
            print(line)
            print(tokenizer.tokenize(line))


PARSER_ENGINE = {
    "main": main,
    "tokenizer": test_tokenizer,
}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("function", default="main")
    # add a positional argument for the function to run

    parser.add_argument(
        "--text", type=str, default="TAF KBLV 010600Z 0106/0212 270020G35KT"
    )
    args = parser.parse_args()
    match args.function:
        case "main":
            main(args.text)
        case "tokenizer":
            test_tokenizer(args.text)
    # PARSER_ENGINE[args.function](args.text)
    # main(args.text)
