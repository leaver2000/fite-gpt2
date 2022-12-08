from typing import Literal, TypedDict, Iterable
import enum

from typing_extensions import Unpack
import dataclasses
import torch
from datasets.dataset_dict import DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments as _TrainingArguments,
    AdamW,
    
)

from ..util import SpecialTokens,  CONSTANTS

# training module imports
from .filesystem import DEFAULT_DEVICE, FileSystem



@dataclasses.dataclass
class TrainingArguments(_TrainingArguments):
    """Training arguments for the Trainer class."""

    overwrite_output_dir = True
    per_device_train_batch_size = CONSTANTS.BATCH_SIZE
    per_device_eval_batch_size = CONSTANTS.BATCH_SIZE
    num_train_epochs = 1
    warmup_steps = 500
    weight_decay = 0.01
    logging_dir = "./logs"
    logging_steps = 10
    save_steps = 1000
    evaluation_strategy = "steps"
    eval_steps = 1000
    load_best_model_at_end = True
    metric_for_best_model = "eval_loss"
    greater_is_better = False
    fp16 = True
    dataloader_num_workers = 0

    @classmethod
    def from_filesystem(cls, fs: FileSystem) -> "TrainingArguments":
        return TrainingArguments(
            output_dir=str(fs.model_path),
            run_name=fs.model_name,
            # do_train=True,
            # do_eval=True,
            # do_predict=True,
        )


def tokenizer(fs: FileSystem) -> None:

    tokenizer = fs.get_pretrained("TOKENIZER")

    # add the additional tokens to the tokenizer from the file system config
    additional_tokens = fs.get("additional-tokens")
    if additional_tokens:
        tokenizer.add_tokens(additional_tokens)  # type: ignore

    special_tokens = SpecialTokens.select("eos_token", "bos_token", "pad_token")

    additional_special_tokens = fs.get("additional-special-tokens")
    if additional_special_tokens:
        special_tokens["additional_special_tokens"] = additional_special_tokens

    tokenizer.add_special_tokens(special_tokens)  # type: ignore

    tokenizer.save_pretrained(fs.tokenizer_path)


class _AdamWOptimizerKwargs(TypedDict, total=False):
    """these are the kwargs for the AdamW optimizer"""

    params: Iterable[torch.nn.parameter.Parameter]
    lr: float  # = 1e-3
    betas: tuple[float, float]  # = (0.9, 0.999)
    eps: float  # = 1e-6
    weight_decay: float  # = 0.0
    correct_bias: bool  # = True
    no_deprecation_warning: bool  # = False


@dataclasses.dataclass
class AdamWOptimizerConfig(dict):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-6
    weight_decay: float = 0.0
    correct_bias: bool = True
    no_deprecation_warning: bool = False

    def pipe(
        self, /, **kwargs: Unpack[_AdamWOptimizerKwargs]
    ) -> "AdamWOptimizerConfig":
        return AdamWOptimizerConfig(**{**self, **kwargs}) 


class OptimizersType(AdamWOptimizerConfig, enum.Enum):
    def __new__(cls, value: AdamWOptimizerConfig):
        obj = {}.__new__(cls)
        obj._value_ = value
        return obj

    BASE_CONFIG = AdamWOptimizerConfig()
    ENCODER = BASE_CONFIG.pipe(lr=5e-5)
    FAST_LEARNER = BASE_CONFIG.pipe(lr=1e-4)
    SLOW_LEARNER = BASE_CONFIG.pipe(lr=1e-5)

@dataclasses.dataclass
class AdamWOptimizer(AdamW):
    params: Iterable[torch.nn.Parameter]
    optimizer_config: AdamWOptimizerConfig
    def __post_init__(self):
        super().__init__(self.params, **self.optimizer_config)




def model(
    fs: FileSystem,
    tokenizer: GPT2TokenizerFast,
    push_to_hub: bool = False,
    framework: Literal["pt"] = "pt",
) -> GPT2LMHeadModel:
    # clear the GPU cache
    torch.cuda.empty_cache()
    # get the pretrained model from the file system and push it to the GPU
    model = fs.get_pretrained("MODEL").to(DEFAULT_DEVICE)
    # resize the embedding layer to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    # load the dataset from the file system
    tokenized_ds = DatasetDict.load_from_disk(str(fs.dataset_dict_path))
    # ###  Trainer Setup ###
    # configure the trainer arguments
    training_arguments = TrainingArguments.from_filesystem(fs)

    # configure the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        # mlm_probability=0.15,
        pad_to_multiple_of=20,
        return_tensors=framework,
    )
    # ###  Trainer Setup ###
    model.train()
    if fs.model_is_local:
        optimizers = (
            AdamWOptimizer(model.parameters(), OptimizersType.ENCODER),
            None,
        )
    else:
        # if the base gpt2 model is used no optimizers are used
        optimizers = (None, None)

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
        optimizers=optimizers,  # type:ignore Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics=None,  # type:ignore (Tensor, Tensor) -> Tensor = None
    )
    # model.forward = trainer.model_wrapped_forward
    # train the model
    trainer.train()
    # save model
    if push_to_hub:
        model.push_to_hub(fs.model_name)
        repo_url = trainer.push_to_hub()
        print(f"Pushed model to 🤗 {repo_url}")

    return model
