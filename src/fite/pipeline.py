import dataclasses
import random
import re
from pathlib import Path
from typing import Iterable, Optional, TypeAlias, TypedDict, overload

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    TextGenerationPipeline,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from typing_extensions import Unpack

from ._enum import DictEnum, StrEnum
from ._typing import ModelConfig, PyProjectTOML

__all__ = [
    "Pipeline",
    "PipelineEngine",
    "Strategies",
    "HyperParameters",
    "HyperParameterStrategy",
]


class Sample(TypedDict):
    do_sample: bool
    num_beams: int
    num_beam_groups: int


class GeneratorKWARGS(TypedDict):
    max_length: Optional[int]
    min_length: Optional[int]
    do_sample: bool
    early_stopping: bool
    num_beams: Optional[int]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    repetition_penalty: Optional[float]
    bad_words_ids: Optional[list[int]]
    bos_token_id: Optional[int]
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    length_penalty: Optional[float]
    no_repeat_ngram_size: Optional[int]
    encoder_no_repeat_ngram_size: Optional[int]
    num_return_sequences: Optional[int]
    max_time: Optional[float]
    max_new_tokens: Optional[int]
    decoder_start_token_id: Optional[int]
    use_cache: bool
    num_beam_groups: Optional[int]
    diversity_penalty: Optional[float]
    # prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]]
    # logits_processor: Optional[LogitsProcessorList]
    # renormalize_logits: bool
    # stopping_criteria: Optional[StoppingCriteriaList]
    # constraints: Optional[list[Constraint]]
    output_attentions: bool
    output_hidden_states: bool
    output_scores: bool
    return_dict_in_generate: bool
    forced_bos_token_id: Optional[int]
    forced_eos_token_id: Optional[int]
    remove_invalid_values: bool
    synced_gpus: bool
    exponential_decay_length_penalty: Optional[tuple[int, float]]
    suppress_tokens: Optional[list[int]]
    begin_suppress_tokens: Optional[list[int]]
    forced_decoder_ids: Optional[list[list[int]]]


class DecoderKWARGS(TypedDict):
    skip_special_tokens: bool
    clean_up_tokenization_spaces: bool


class HyperParameters(TypedDict, total=False):
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
)


class HyperParameterStrategy(DictEnum):
    GREEDY = HyperParameters(do_sample=False, num_beams=1)
    """
    https://huggingface.co/blog/how-to-generate#greedy-search

    Greedy search simply selects the word with the highest probability as its next word
    """
    BEAM_SEARCH = HyperParameters(do_sample=False, num_beams=5, early_stopping=True)
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

    # --- Top-K + Top-p (nucleus) sampling ---
    TOP_KP = TOP_K50 | TOP_P92
    TOP_KP_T125 = TOP_KP | TEMP_125
    TOP_KP_T150 = TOP_KP | TEMP_150
    TOP_KP_T175 = TOP_KP | TEMP_175


# kinda of a hacky solution to dynamically add the strategies to the class and keep the linter happy
# otherwise the class is represented as a type[Enum] and the linter complains about the @classmethod's
_Strategies: type[StrEnum] = StrEnum("Strategies", HyperParameterStrategy.list_members())  # type: ignore
Strategies: TypeAlias = _Strategies


class TokenIds(TypedDict):
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None


def strip_split(text_string: str) -> list[str]:
    return [
        string.strip() for string in re.split(r"(?=BECMG|TEMPO)", text_string) if string
    ]


@dataclasses.dataclass
class Pipeline(TextGenerationPipeline):
    """
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/model#generative-models
    """

    model: GPT2LMHeadModel
    tokenizer: GPT2Tokenizer | GPT2TokenizerFast
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
        return [strip_split(text.replace("\n", "")) for text in text_list]

    @overload
    def generate(
        self,
        text: str,
        strategy: Optional[HyperParameterStrategy | Strategies | str] = None,
        padding: PaddingStrategy | bool = True,
        truncation: TruncationStrategy | bool = True,
        **kwargs: Unpack[HyperParameters],
    ) -> list[str]:
        ...

    @overload
    def generate(
        self,
        text: list[str],
        strategy: Optional[HyperParameterStrategy | Strategies | str] = None,
        padding: PaddingStrategy | bool = True,
        truncation: TruncationStrategy | bool = True,
        **kwargs: Unpack[HyperParameters],
    ) -> list[list[str]]:
        ...

    def generate(
        self,
        text: str | list[str],
        strategy: Optional[HyperParameterStrategy | Strategies | str] = None,
        padding: PaddingStrategy | bool = True,
        truncation: TruncationStrategy | bool = True,
        **kwargs: Unpack[HyperParameters],
    ) -> list[list[str]] | list[str]:
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

        if not isinstance(strategy, HyperParameterStrategy):
            strategy = HyperParameterStrategy[
                strategy
                if strategy
                else random.choice(HyperParameterStrategy.list_members())
            ]
            # type: ignore

        # BatchEncoding provides the token_ids and attention_mask
        encoding = self.encode(text, padding=padding, truncation=truncation)
        # update the encoding with the token_ids, hyper-parameters and any kwargs
        encoding |= token_ids | strategy | kwargs
        # call the model to generate by unpacking all of the contents of the encoding
        outputs = model.generate(**encoding, max_length=self.max_length)
        # decode the batch results
        result = self.batch_decode(outputs)
        if len(result) == 1:
            (result,) = result
        return result

    @property
    def token_ids(self) -> TokenIds:

        return TokenIds(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )


@dataclasses.dataclass
class PipelineEngine:
    root_path: Path | str
    models: list[ModelConfig]
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu", index=0
    )

    def __post_init__(self):
        root = self.root_path
        if isinstance(root, str):
            root = Path(root)

        self.__pipelines: dict[str, Pipeline] = {}
        for model in self.models:
            pipeline_path = root / model["model-name"]
            model_path = pipeline_path / "model"
            tokenizer_path = pipeline_path / "tokenizer"
            if not model_path.exists() or not tokenizer_path.exists():
                import warnings  # noqa

                warnings.warn(
                    f"Model {model['model-name']} was found in the config but not on disk\n"
                    f"Model path: {model_path}\n"
                    f"Tokenizer path: {tokenizer_path}\n"
                    "Skipping..."
                )

                continue

            self.__pipelines[model["model-name"]] = Pipeline(
                model=GPT2LMHeadModel.from_pretrained(model_path),  # type: ignore
                tokenizer=GPT2Tokenizer.from_pretrained(tokenizer_path),
                device=self.device,
                max_length=256,
            )

    @classmethod
    def load_from_pyproject(
        cls, path: str | Path = "pyproject.toml"
    ) -> "PipelineEngine":
        return cls.from_pyproject(PyProjectTOML.load(path))

    @classmethod
    def from_pyproject(cls, pyproject: PyProjectTOML) -> "PipelineEngine":
        config = pyproject["tool"]["fite"]
        return PipelineEngine(root_path=config["root-path"], models=config["models"])

    def list_models(self) -> list[str]:
        return list(self.__pipelines.keys())

    def get_pipeline(self, name: str | StrEnum) -> Pipeline:
        return self.__pipelines[name]

    def get_model(self, name: str | StrEnum) -> GPT2LMHeadModel:
        return self.get_pipeline(name).model

    @overload
    def generate(
        self,
        pipeline: str | StrEnum,
        text: list[str],
        strategy: str | StrEnum | None = None,
    ) -> list[list[str]]:
        ...

    @overload
    def generate(
        self,
        pipeline: str | StrEnum,
        text: str,
        strategy: str | StrEnum | None = None,
    ) -> list[str]:
        ...

    def generate(
        self,
        pipeline: str | StrEnum,
        text: str | list[str],
        strategy: str | StrEnum | None = None,
    ) -> list[list[str]] | list[str]:
        return self.get_pipeline(pipeline).generate(text, strategy=strategy)
