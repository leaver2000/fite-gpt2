import dataclasses
import enum
import os
import sys
from contextlib import contextmanager, redirect_stdout
from typing import Generic, Union

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

import torch
from transformers import AddedToken

from ._enum import RegexEnum, StrEnum
from .pipeline import HyperParameters, HyperParameterStrategy

__all__ = [
    "ResultRecord",
    "SpecialTokens",
    "ActionStr",
    "CONSTANTS",
    "GPT2BaseModels",
]
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

Ts = TypeVarTuple("Ts")


class CONSTANTS(enum.IntEnum):
    BATCH_SIZE = os.getenv("BATCH_SIZE", 8)
    MAX_LENGTH = os.getenv("MAX_LENGTH", 256)
    NUM_PROC = os.getenv("NUM_PROC", 4)


@dataclasses.dataclass
class DataclassBase(Generic[Unpack[Ts]]):
    """
    Generic dataclass base class that allows for easy conversion to a dictionary.
    """

    def to_dict(self) -> dict[str, Union[Unpack[Ts]]]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ResultRecord(
    DataclassBase[str, list[str], float, HyperParameters | HyperParameterStrategy]
):
    """ResultOutput is a named tuple that contains the output of the model."""

    model: str
    prompt_text: str
    generated_text: list[str]
    strategy: str
    hyper_parameters: HyperParameterStrategy | HyperParameters

    def evaluate(self):
        self.score = 0.0

        return self


AdditionalSpecialTokens = list[str] | list[AddedToken] | list[str | AddedToken]
SpecialTokenDict = dict[str, AddedToken | str | AdditionalSpecialTokens]


class TokenEnum(RegexEnum):
    value: str

    @classmethod
    def to_dict(cls) -> SpecialTokenDict:
        return {k: v.value for k, v in cls._member_map_.items()}

    @classmethod
    def with_additional(cls, tokens: AdditionalSpecialTokens) -> SpecialTokenDict:
        special_tokens = cls.to_dict().copy()
        if tokens:
            special_tokens["additional_special_tokens"] = tokens
        return special_tokens

    @classmethod
    def select(cls, *names) -> SpecialTokenDict:
        return {name: cls[name].value for name in names}


class SpecialTokens(TokenEnum):
    """Special tokens that can be added to a tokenizer"""

    bos_token = "<|bos|>"
    eos_token = "<|eos|>"
    # pad_token = "[pad]"
    # cls_token = "<|cls|>"
    pad_token = "<|pad|>"
    unk_token = "<|unk|>"  # may be useful to mask missing header tokens
    # metadata = "[metadata]"


class ActivationFunctions(StrEnum):
    """Activation functions"""

    relu = enum.auto()
    """
    `ReLU` - Rectified Linear Unit

    `f(x) = max(0, x)`

    RelU does not activate all neurons at the same time, the neurons will only be activated if
    the output of the previous layer is less than 0.
    """
    silu = enum.auto()
    gelu = enum.auto()
    """
    `GELU` - Gaussian Error Linear Unit

    `f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`

    GELU is a smooth approximation of the ReLU function and allows for better back-propagation.
    
    """
    gelu_new = enum.auto()
    tanh = enum.auto()


class ActionStr(StrEnum):
    """ParserActions are the actions that the parser can take.
    >>> ParserActions.HELP
    'help'
    >>> ParserActions.STORE
    'store'
    """

    STORE = "store"
    CONST = "store_const"
    TRUE = "store_false"
    FALSE = "store_true"
    APPEND = "append"
    APPEND_CONST = "append_const"
    COUNT = "count"
    HELP = "help"
    VERSION = "version"


class GPT2BaseModels(StrEnum):
    """GPT2BaseModels are the base models that can be used with the GPT2 model."""

    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield
