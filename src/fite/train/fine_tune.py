from typing import Literal

import torch
from datasets.dataset_dict import DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

from ..util import CONSTANTS, DEFAULT_DEVICE, FileSystem, SpecialTokens


def tokenizer(fs: FileSystem) -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained(
        fs.base_model,
        num_labels=2,
        do_basic_tokenize=False,
    )

    # #  add the additional tokens to the tokenizer from the file system config
    tokenizer.add_tokens(fs.config["additional-tokens"])  # type: ignore

    special_tokens = SpecialTokens.select("eos_token", "bos_token", "pad_token")
    special_tokens["additional_special_tokens"] = fs.config["additional-special-tokens"]
    tokenizer.add_special_tokens(special_tokens)  # type: ignore

    tokenizer.save_pretrained(fs.tokenizer)


def model(
    fs: FileSystem,
    tokenizer: GPT2TokenizerFast,
    push_to_hub: bool = False,
    framework: Literal["pt"] = "pt",
) -> None:
    torch.cuda.empty_cache()
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
        fs.base_model,
        # GPT2Config -> https://huggingface.co/transformers/v3.5.1/model_doc/gpt2.html#gpt2config
        config=GPT2Config(
            activation_function="gelu_new",  # ["relu", "silu", "gelu", "tanh", "gelu_new"]
            layer_norm_eps=1e-05,
        ),
    ).to(  # type: ignore
        DEFAULT_DEVICE
    )
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))

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
        return_tensors=framework,
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
