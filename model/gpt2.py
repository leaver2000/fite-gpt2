# standard library imports
"""
https://huggingface.co/transformers/v2.0.0/examples.html#gpt-2-gpt-and-causal-language-modeling


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

``` bash
pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

"""
import os
import re

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
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from transformers import TextGenerationPipeline

# from tokenizers.implementations import BertWordPieceTokenizer
# datasets imports
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset, Batch

from .dataset import JSONLines
from .util import (
    SpecialTokens,
    CodePredictionPipeline,
    unpack_paths,
    get_raw_text_data,
    train_test_split,
    JSON_LINES_FILE,
)

# RUNTIME VARIABLES
VERSION = "0.0.3dev-8"
PRE_TRAINED_MODEL_NAME = "gpt2"
DATASET_PREP_METHOD = "taf-completion"
FRAMEWORK = "pt"
MODEL_NAME = f"{PRE_TRAINED_MODEL_NAME}-{DATASET_PREP_METHOD}"
MODEL_PATH, DATASET_PATH = unpack_paths(MODEL_NAME, VERSION)
BATCH_SIZE = 8
# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

# Hyper-parameters
REPETITION_PENALTY = 1.8
"repetition penalty is a hyperparameter that controls the model's repetition of the same token. The higher the value, the more repetitive the text will be."
TEMPERATURE = 1.0
"temperature is a hyperparameter that controls the randomness of the model's predictions. The higher the value, the more random the text will be."
TOP_K = 2
"top_k is a hyperparameter that controls the number of tokens that the model will consider when predicting the next token. The higher the value, the more random the text will be."
TOP_P = 0.9
"top_p is a hyperparameter that controls the number of tokens that the model will consider when predicting the next token. The higher the value, the more random the text will be."
MAX_LENGTH = 120
"max_length is a hyperparameter that controls the maximum length of the generated text. The higher the value, the longer the text will be."


vocab_dict = {}


# load tokenizer
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
    PRE_TRAINED_MODEL_NAME,
    # TODO: adding a vocab file to the tokenizer
    # vocab_file=os.path.join(MODEL_PATH, "vocab.json"),
)
tokenizer.add_special_tokens(SpecialTokens.to_dict())


def create_tokenized_dataset() -> DatasetDict:
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
    return ds


def fine_tune_model() -> None:
    torch.cuda.empty_cache()
    # base model
    configuration = GPT2Config(
        activation_function="gelu_new",
        layer_norm_eps=1e-05,
    )
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        config=configuration,
    ).cuda(  # type: ignore
        device
    )
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    # tokenized dataset
    if not DATASET_PATH.exists():
        # if dataset does not exist, create it
        (
            JSONLines.load(
                JSON_LINES_FILE,
                strip_temps=True,
                drop_wnd_aft_rmks=True,
            )
            .to_dataset_dict()
            .map(
                lambda x: tokenizer(x["prompt"], truncation=True, padding=True),
                batched=True,
                batch_size=BATCH_SIZE,
            )
            .map(
                lambda x: tokenizer(x["completion"], truncation=True, padding=True),
                batched=True,
                batch_size=BATCH_SIZE,
            )
            .save_to_disk(DATASET_PATH)  # type: ignore
        )
    # ###  Trainer Setup ###
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
        return_tensors=FRAMEWORK,
        pad_to_multiple_of=3,
    )
    # create trainer
    trainer = Trainer(
        model=model.train(),
        tokenizer=tokenizer,
        train_dataset=tokenized_ds["train"],  # type: ignore
        eval_dataset=tokenized_ds["validation"],  # type: ignore
        args=training_arguments,
        data_collator=data_collator,
        # compute_metrics=lambda p: {"loss": p["loss"]},
    )
    # train model
    trainer.train()
    # save model
    model.save_pretrained(MODEL_PATH)  # type: ignore


if not MODEL_PATH.exists():
    # if a new model is being trained fine tune the model
    fine_tune_model()
# load the model from the saved path
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).cuda(device)  # type: ignore

pipeline = CodePredictionPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    temperature=TEMPERATURE,
    max_length=MAX_LENGTH,
)


def main(text: str) -> None:
    """
    ### example:
    provided the string below the model correctly encoded the temporary group by including a visibility obstruction and a lower ceiling.
    this is a very common use case when encoding TEMPO groups during showers

    The third line in the taf has a-lot more randomness to it.

    >>> python -m model.gpt2 --text "TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS\\nTEMPO 0200/0206 5000"
    [[{'generated_text': 'TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS\\nTEMPO 0200/0206 5000 BR BKN015\\nBECMG 0314 VRG18650 510004 510013 650726 521044 510353 SN SCT024 620303 530154 540403 FEW017 VCSH 0512Z 54 01006W 4800 RA SKC WS009CB 56012QLD035 520204 FG 9000 DVRS 621958 610002 623504 3'}]]


    """

    print(pipeline.predict(text.split("\n")))


def test_tokenizer(file_in="store/training-data-v2.jsonl") -> None:
    ds = JSONLines.load(file_in).to_dataset_dict()

    ds = ds.map(
        lambda x: tokenizer(x["prompt"], truncation=True, padding=True),
        batched=True,
        batch_size=BATCH_SIZE,
    ).map(
        lambda x: tokenizer(x["completion"], truncation=True, padding=True),
        batched=True,
        batch_size=BATCH_SIZE,
    )


def generate_json_lines(
    file_in="store/training-data-v2.txt", file_out="store/training-data-v2.jsonl"
):
    # NOTE: this is a slow function but should only be run once
    # when the dataset is first created.
    import json
    from .dataset import TAFDataset

    with open(file_in, "r") as f:
        taf = TAFDataset(f.read().split("\n\n###\n\n"))

    # two variations of the same dataset are loaded
    taf_list = [
        # the first dataset split every new line in the taf which is then split again into
        # a prompt and a completion
        taf.to_json_lines(split_each_line=True).to_frame()
        for _ in range(1_000)
    ] + [
        # the second dataset splits each TAF into a prompt and a completion
        # there is a much lower chance of the model seeing the same prompt-completion pair
        # so the range is much higher to create a larger dataset
        taf.to_json_lines(split_each_line=False).to_frame()
        for _ in range(5_000)
    ]
    # combine the two datasets and drop duplicates
    df = pd.concat(taf_list).drop_duplicates().reset_index()
    # remove and whitespace from the prompt and completion
    df = df.drop(df.index[df.completion == ""])
    time_cols = ["issue_time", "from_valid_time", "to_valid_time"]
    # format the time columns
    df[time_cols] = df[time_cols].stack().dt.strftime("%Y-%m-%dT%H:%M:%SZ").unstack()

    with open(file_out, "w") as f:
        for record in df.to_dict(orient="records"):
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", type=str, default="TAF KBLV 010600Z 0106/0212 270020G35KT"
    )
    args = parser.parse_args()
    main(args.text)