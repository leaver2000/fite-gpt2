# standard library imports
"""
# gpt2-taf

## Description

This is a project to train the GPT-2 model for TAF autocompletion.
provided with a partial TAF, the model will attempt to generate the rest of the TAF.

GPT-2 uses multi-layer transformers to generate text.

## Drawbacks

GPT-2 are that is is uni-directional and can only complete left to right.
with a TAF this presents some challenges as the max and min temperature group come at
the end of the forecast.  The model may be able to better resolve present weather
predictions if the max and min temperature are provided first.


## TODO:
- [ ] increase the size of the training dataset
- [ ] add labels to the dataset
- [ ] pad tokens into the dataset
- [ ] add a classification head to the model
- [ ] add a regression head to the model
- [ ] add a sequence classification head to the model
- [ ] add a token classification head to the model


references:
https://medium.com/@gauravghati/comparison-between-bert-gpt-2-and-elmo-9ad140cd1cda
https://huggingface.co/transformers/model_doc/gpt2.html


``` bash
python -m torch.utils.collect_env

Collecting environment information...
PyTorch version: 1.13.0+cu117
Is debug build: False
CUDA used to build PyTorch: 11.7
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.2.0-19ubuntu1) 11.2.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.6 (main, Nov  2 2022, 18:53:38) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.10.102.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.5.119
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 2080 SUPER
Nvidia driver version: 516.59
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.2.4
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] mypy-extensions==0.4.3
[pip3] numpy==1.23.5
[pip3] torch==1.13.0+cu117
[conda] Could not collect
```

A specific version of pytorch was required for my build of the model

``` bash
pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```
"""
import os

# keep tensorflow quiet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# general imports
import torch
import pandas as pd

# transformer imports
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from transformers import Trainer, TrainingArguments
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from transformers import TextGenerationPipeline

# datasets imports
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset, Batch

from .util import (
    SpecialTokens,
    unpack_paths,
    get_raw_text_data,
    train_test_split,
)

# RUNTIME VARIABLES
VERSION = "0.0.2"
PRE_TRAINED_MODEL_NAME = "gpt2"
DATASET_PREP_METHOD = "taf-full"
PYTORCH_FRAMEWORK = "pt"
MODEL_NAME = f"{PRE_TRAINED_MODEL_NAME}-{DATASET_PREP_METHOD}"
MODEL_PATH, DATASET_PATH = unpack_paths(MODEL_NAME, VERSION)
BATCH_SIZE = 8
MAX_LENGTH = 120
# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
# load tokenizer
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer.add_special_tokens(SpecialTokens.to_dict())


def create_tokenized_dataset() -> None:
    """Create a tokenized dataset from the raw text data"""

    def batch_encode(batch: Batch) -> BatchEncoding:
        return tokenizer(batch["text"], truncation=True)  # type: ignore

    text = get_raw_text_data()
    lines = text.split("\n\n###\n\n")
    assert all(len(line) > 0 for line in lines)
    df = (
        pd.Series(lines, name="text")
        .where(lambda s: s != "")
        .dropna()
        .reset_index(drop=True)
        .to_frame()
    )
    # create labels for classification
    # TODO: add labels to the dataset
    # df["labels"] = np.where(df.text.str.startswith("TAF"), 1, 0)
    # TODO: pad tokens into the dataset
    # df["text"] = SpecialTokens.bos_token + df.text.str.replace("\n", SpecialTokens.sep_token) + SpecialTokens.eos_token
    # split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2)
    # create a DatasetDict
    ds = DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        test=Dataset.from_pandas(test_df, preserve_index=False),
    )
    # tokenize dataset with batch encoding using the tokenizer
    ds = ds.map(batch_encode, batched=True, batch_size=BATCH_SIZE)
    ds.save_to_disk(DATASET_PATH)  # type: ignore


def fine_tune_model() -> None:
    """Fine tune the model on the tokenized dataset"""
    torch.cuda.empty_cache()
    # gpt2 config
    configuration = GPT2Config(
        activation_function="gelu_new",
        layer_norm_eps=1e-05,
    )
    # load base model
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        config=configuration,
    ).cuda(  # type: ignore
        device
    )
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    if not DATASET_PATH.exists():
        # if dataset does not exist, create it
        create_tokenized_dataset()
    # load the dataset
    tokenized_ds = DatasetDict.load_from_disk(DATASET_PATH)  # type: ignore
    # configure the trainer arguments
    training_arguments = TrainingArguments(
        run_name=MODEL_NAME,
        output_dir=MODEL_PATH,  # type: ignore
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
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
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.15,
        return_tensors=PYTORCH_FRAMEWORK,
        pad_to_multiple_of=3,
    )
    # create trainer
    trainer = Trainer(
        model=model.train(),
        train_dataset=tokenized_ds["train"],  # type: ignore
        eval_dataset=tokenized_ds["test"],  # type: ignore
        args=training_arguments,
        data_collator=data_collator,
        # compute_metrics=lambda p: {"loss": p["loss"]},
    )
    # train model
    trainer.train()
    # save model
    model.save_pretrained(MODEL_PATH)  # type: ignore


def main(text: str) -> None:
    if not MODEL_PATH.exists():
        # if a new model is being trained fine tune the model
        fine_tune_model()
    # load the model from the saved path
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).cuda(device)  # type: ignore

    pipe = TextGenerationPipeline(
        model=model,  # PreTrainedModel
        tokenizer=tokenizer,  # PreTrainedTokenizer
        device=device,  # torch.device
        PYTORCH_FRAMEWORK=PYTORCH_FRAMEWORK,  # Literal["pt", "tf"]
        temperature=0.2,  # strictly positive float
        top_k=100,  # int
        top_p=0.1,  # float
        # repetition_penalty=2.0,  # float
        # do_sample=True,  # bool
    )

    print(pipe(text, max_length=MAX_LENGTH))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", type=str, default="TAF KBLV 010600Z 0106/0212 270020G35KT"
    )
    args = parser.parse_args()
    main(args.text)
