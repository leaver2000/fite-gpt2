from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch

from . import __version__ as VERSION
from .util import (
    get_model_name,
    get_paths,
    DatasetNames,
    BaseModelNames,
    MAX_LENGTH,
    DEFAULT_DEVICE,
)
from ._pipeline import CodePredictionPipeline

__all__ = ["pipeline"]


def pipeline(
    base_model: BaseModelNames,
    dataset_name: DatasetNames,
    version: str = VERSION,
    device: torch.device = DEFAULT_DEVICE,
    num_return_sequences: int = 1,
) -> CodePredictionPipeline:
    model_name = get_model_name(base_model, dataset_name, version)
    model_path, tokenizer_path, _ = get_paths(model_name)

    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_path)  # type: ignore
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

    return CodePredictionPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=MAX_LENGTH,
        num_return_sequences=num_return_sequences,
    )
