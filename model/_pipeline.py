import re
import enum
import dataclasses
from typing import TypedDict, TypeAlias, Optional, Literal, Iterable

from typing_extensions import Unpack

import torch
from transformers import (
    GPT2LMHeadModel,
    TextGenerationPipeline,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from .typing import (
    GPT2TokenizerType,
)

__all__ = ["CodePredictionPipeline", "HyperParameters", "HyperParameterStrategy"]
StandardTokens = Literal[
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "sep_token",
    "additional_special_tokens",
]
WS_OPT = r"(?:\s+)?"  # optional whitespace
uint: TypeAlias = int
UnsignedFloat: TypeAlias = float


class Sample(TypedDict):
    do_sample: bool
    num_beams: int
    num_beam_groups: int


class _GeneratedText(TypedDict):
    generated_text: str


GeneratedText = dict[str, str] | _GeneratedText
GenerationStrategy = Literal["greedy", "beam_search", "sample", "top_k", "top_p"]


class GeneratorKWARGS(TypedDict):
    max_length: Optional[uint]
    min_length: Optional[uint]
    do_sample: bool
    early_stopping: bool
    num_beams: Optional[uint]
    temperature: Optional[UnsignedFloat]
    top_k: Optional[uint]
    top_p: Optional[UnsignedFloat]
    repetition_penalty: Optional[UnsignedFloat]
    bad_words_ids: Optional[list[int]]
    bos_token_id: Optional[uint]
    pad_token_id: Optional[uint]
    eos_token_id: Optional[uint]
    length_penalty: Optional[UnsignedFloat]
    no_repeat_ngram_size: Optional[uint]
    encoder_no_repeat_ngram_size: Optional[uint]
    num_return_sequences: Optional[uint]
    max_time: Optional[UnsignedFloat]
    max_new_tokens: Optional[uint]
    decoder_start_token_id: Optional[uint]
    use_cache: bool
    num_beam_groups: Optional[uint]
    diversity_penalty: Optional[UnsignedFloat]
    # prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]]
    # logits_processor: Optional[LogitsProcessorList]
    # renormalize_logits: bool
    # stopping_criteria: Optional[StoppingCriteriaList]
    # constraints: Optional[list[Constraint]]
    output_attentions: bool
    output_hidden_states: bool
    output_scores: bool
    return_dict_in_generate: bool
    forced_bos_token_id: Optional[uint]
    forced_eos_token_id: Optional[uint]
    remove_invalid_values: bool
    synced_gpus: bool
    exponential_decay_length_penalty: Optional[tuple[uint, UnsignedFloat]]
    suppress_tokens: Optional[list[int]]
    begin_suppress_tokens: Optional[list[int]]
    forced_decoder_ids: Optional[list[list[int]]]


class DecoderKWARGS(TypedDict):
    skip_special_tokens: bool
    clean_up_tokenization_spaces: bool


class HyperParameters(TypedDict, total=False):
    # max_length: int | None = None, min_length: int | None = None,
    # do_sample: bool | None = None,
    # early_stopping: bool | None = None,
    # num_beams: int | None = None,
    do_sample: bool
    num_beams: int
    num_beam_groups: int
    num_return_sequences: int
    no_repeat_ngram_size: int
    # temperature: float | None = None,
    temperature: Optional[float]
    # penalty_alpha: float | None = None,
    penalty_alpha: Optional[float]
    # top_k: int | None = None,
    top_k: Optional[int]
    # top_p: float | None = None,
    top_p: Optional[float]
    # typical_p: float | None = None,
    typical_p: Optional[float]
    # repetition_penalty: float | None = None
    repetition_penalty: Optional[float]
    early_stopping: bool
    no_repeat_ngram_size: int
    num_beams: int


_BASE_SAMPLE = HyperParameters(
    do_sample=True,
    top_k=0,
    num_beams=1,
    # num_return_sequences=5,
)


class HyperParameterStrategy(dict, enum.Enum):
    value: HyperParameters
    GREEDY = HyperParameters(
        do_sample=False,
        num_beams=1,
    )
    """
    https://huggingface.co/blog/how-to-generate#greedy-search

    Greedy search simply selects the word with the highest probability as its next word
    """
    BEAM_SEARCH = HyperParameters(
        do_sample=False,
        num_beams=5,
        early_stopping=True,
    )
    """
    https://huggingface.co/blog/how-to-generate#beam-search

    Beam search reduces the risk of missing hidden high probability
    word sequences by keeping the most likely num_beams of hypotheses
    at each time step and eventually choosing the hypothesis that has the
    overall highest probability. Let's illustrate with num_beams=2:
    """
    BEAM_SEARCH_N_GRAM = BEAM_SEARCH | HyperParameters(no_repeat_ngram_size=2)
    """
    https://arxiv.org/abs/1705.04304

    In transformers, we simply set the parameter num_return_sequences to the number of
    highest scoring beams that should be returned.
    Make sure though that num_return_sequences <= num_beams!
    """
    BEAM_SEARCH_N_GRAM_5 = BEAM_SEARCH_N_GRAM | HyperParameters(num_return_sequences=5)

    # --- SAMPLING ---
    # In its most basic form, sampling means randomly picking the next word
    # according to its conditional probability distribution

    # use temperature to decrease the sensitivity to low probability candidates
    # NOTE: a temperature of less than 8.5 and above 2.0 tends to produce a lot of gibberish
    TEMP_085 = _BASE_SAMPLE | HyperParameters(temperature=0.85)
    TEMP_095 = _BASE_SAMPLE | HyperParameters(temperature=0.95)
    TEMP_105 = _BASE_SAMPLE | HyperParameters(temperature=1.05)
    TEMP_125 = _BASE_SAMPLE | HyperParameters(temperature=1.25)
    TEMP_150 = _BASE_SAMPLE | HyperParameters(temperature=1.5)
    TEMP_175 = _BASE_SAMPLE | HyperParameters(temperature=1.75)
    TEMP_200 = _BASE_SAMPLE | HyperParameters(temperature=2.0)

    # --- Top-K Sampling ---
    # Top-K sampling is a more sophisticated version of sampling where the
    # K most likely next words are filtered and the probability mass is redistributed
    # only among those K next words.
    TOP_K5 = _BASE_SAMPLE | HyperParameters(top_k=5)
    TOP_K50 = _BASE_SAMPLE | HyperParameters(top_k=50)
    TOP_K100 = _BASE_SAMPLE | HyperParameters(top_k=100)
    TOP_K50_T125 = TOP_K50 | TEMP_125
    TOP_K150_T150 = TOP_K50 | TEMP_150
    TOP_K175_T175 = TOP_K50 | TEMP_175
    # --- Top-p (nucleus) sampling ---
    #  https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling
    # Instead of sampling only from the most likely K words,
    # in Top-p sampling chooses from the smallest possible set
    # of words whose cumulative probability exceeds the probability p.
    # The probability mass is then redistributed among this set of words.
    # This way, the size of the set of words (a.k.a the number of words in the set)
    # can dynamically increase and decrease according to the next word's
    # probability distribution. Ok, that was very wordy, let's visualize.
    TOP_P88 = _BASE_SAMPLE | HyperParameters(top_p=0.88)
    TOP_P92 = _BASE_SAMPLE | HyperParameters(top_p=0.92)
    TOP_P96 = _BASE_SAMPLE | HyperParameters(top_p=0.96)
    TOP_P98 = _BASE_SAMPLE | HyperParameters(top_p=0.98)
    TOP_P99 = _BASE_SAMPLE | HyperParameters(top_p=0.99)
    TOP_P200 = _BASE_SAMPLE | HyperParameters(top_p=2.0)
    TOP_P92_T125 = TOP_P92 | TEMP_125
    TOP_P92_T150 = TOP_P92 | TEMP_150
    TOP_P92_T175 = TOP_P92 | TEMP_175
    # TOP_P125 = TOP_P | TEMP_125
    # TOP_P150 = TOP_P | TEMP_150
    # TOP_P175 = TOP_P | TEMP_175
    # # ### Top-p (nucleus) sampling
    TOP_KP = TOP_K50 | TOP_P92
    TOP_KP_T125 = TOP_KP | TEMP_125
    TOP_KP_T150 = TOP_KP | TEMP_150
    TOP_KP_T175 = TOP_KP | TEMP_175


class TokenIds(TypedDict):
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None


def strip_split(text_string: str) -> list[str]:
    return [
        string.strip() for string in re.split(r"(?=BECMG|TEMPO)", text_string) if string
    ]


@dataclasses.dataclass
class CodePredictionPipeline(TextGenerationPipeline):
    """
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/model#generative-models
    """

    model: GPT2LMHeadModel
    tokenizer: GPT2TokenizerType
    device: torch.device
    max_length: int
    num_return_sequences: int = 3
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    num_workers: int = 5
    length_penalty: float = 1.0
    binary_output: bool = False
    use_cache: bool = True
    clean_up_tokenization_spaces: bool = True

    def __post_init__(self):
        config = dataclasses.asdict(self) | self.token_ids
        if self.num_return_sequences > 1:
            config |= Sample(
                do_sample=False,
                num_beams=3,
                num_beam_groups=3,
            )

        super().__init__(task="text-generation", **config)

    def forecast(
        self,
        text: str,
        # max_length=120,
        num_beams=3,
        temperature: float = 0.1,  # randomness -> The higher the temperature, the more random the text will be.
        top_k: int = 1,  # top-k-filtering -> number of highest probability vocabulary tokens to keep.
        #  Between 1 and infinity.
        top_p: float = 0.2,  # top-p-filtering -> probability of keeping tokens. Between 0 and 1.
        repetition_penalty=0.2,  # strictly positive float
        length_penalty=1.0,
        early_stopping=True,
        num_return_sequences=1,
        #  temperature=0.9, top_k=10, top_p=0.92, num_return_sequences=1
    ):

        return self(
            text,
            max_length=self.max_length,
            num_beams=num_beams,
            # num_beam_groups=num_beams,  # `num_beams` should be divisible by `num_beam_groups` for group beam search.
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            # clean_up_tokenization_spaces=False,
        )

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        padding: PaddingStrategy | bool = True,
        truncation: TruncationStrategy | bool = True,
        #
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> BatchEncoding:

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,  # default = True
            padding=padding,  # default = False
            truncation=truncation,  # default = False
            max_length=self.max_length,  # default = None
            stride=0,  # default = 0
            is_split_into_words=False,  # default = False
            pad_to_multiple_of=None,  # default = None
            return_tensors=self.framework,  # default = None
            return_token_type_ids=None,  # default = None
            return_attention_mask=None,  # default = None
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            #
            verbose=verbose,
            **kwargs,
        ).to(self.device)

    def batch_decode(
        self,
        token_ids: torch.Tensor | Iterable[torch.Tensor],
    ) -> list[list[str]]:

        text_list = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # type: ignore

        return [strip_split(text.replace("\n", "")) for text in text_list]

    def generate_forecast(
        self,
        text: str | list[str],
        padding: PaddingStrategy | bool = True,
        truncation: TruncationStrategy | bool = True,
        strategy: HyperParameterStrategy | str | None = None,
        **kwargs: Unpack[HyperParameters],
    ) -> list[list[str]]:
        """
        The hyperparameter strategy is used to generate the hyper parameters for the model.
        they can be overwritten by kwargs.
        """
        # unpack some variables
        model, token_ids = self.model, self.token_ids
        if isinstance(text, list):
            text = [t.strip() for t in text]
            self.tokenizer.padding_side = "left"
        else:
            text = text.strip()
            self.tokenizer.padding_side = "right"

        if isinstance(strategy, str):
            strategy = HyperParameterStrategy[strategy]

        # batch_encoding provides the token_ids and attention_mask
        encoding = self.encode(
            text,
            padding=padding,
            truncation=truncation,
        )
        outputs = model.generate(
            **encoding,
            **token_ids,
            **(strategy if strategy else {}),
            **kwargs,
            max_length=self.max_length,
        )
        # return outputs
        return self.batch_decode(outputs)

    @property
    def token_ids(self) -> TokenIds:

        return TokenIds(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
