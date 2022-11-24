import dataclasses

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
from transformers.pipelines import (
    ArgumentHandler,
    PipelineDataFormat,
    PipelineException,
)

__all__ = [
    "CodePredictionPipeline",
]
from typing import TypedDict


class Sample(TypedDict):
    do_sample: bool
    num_beams: int
    num_beam_groups: int


@dataclasses.dataclass
class CodePredictionPipeline(TextGenerationPipeline):
    """
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/model#generative-models
    """

    model: GPT2LMHeadModel
    tokenizer: GPT2Tokenizer
    device: torch.device
    repetition_penalty: float
    temperature: float
    top_k: int
    top_p: float
    max_length: int
    num_return_sequences: int = 3
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    num_workers: int = 5
    length_penalty: float = 1.0
    binary_output: bool = False  # Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
    use_cache: bool = True
    # sampling
    # do_sample: bool = True
    # diversity_penalty: float = 2.0
    # num_beams: int = 3
    # num_beam_groups: int = 3
    # sample :bool = True
    clean_up_tokenization_spaces: bool = True

    def __post_init__(self):
        config = dataclasses.asdict(self) | self.token_ids()

        if self.num_return_sequences > 1:
            config |= Sample(
                do_sample=False,
                num_beams=3,
                num_beam_groups=3,
            )
        # argument signature passed to parent class
        # super().__init__(
        #   model: Union["PreTrainedModel", "TFPreTrainedModel"],
        #   tokenizer: Optional[PreTrainedTokenizer] = None,
        #   feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        #   modelcard: Optional[ModelCard] = None,
        #   framework: Optional[str] = None,
        #   task: str = "",
        #   args_parser: ArgumentHandler = None,
        #   device: Union[int, str, "torch.device"] = -1,
        #   binary_output: bool = False,
        #   **kwargs
        # )
        super().__init__(task="text-generation", **config)

    def token_ids(self) -> dict[str, int | None]:
        # ValueError: The following `model_kwargs` are not used by the model: ['sep_token_id', 'cls_token_id']
        # "cls_token_id",
        # "sep_token_id",
        token_id_keys = {
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
        }
        key_map = {key: getattr(self.tokenizer, key, None) for key in token_id_keys}

        return key_map  # type: ignore

    def forecast(
        self,
        text: str,
        # max_length=120,
        num_beams=3,
        temperature: float = 0.1,  # randomness -> The higher the temperature, the more random the text will be.
        top_k: int = 1,  #  top-k-filtering -> number of highest probability vocabulary tokens to keep. Between 1 and infinity.
        top_p: float = 0.2,  # top-p-filtering -> probability of keeping tokens. Between 0 and 1.
        repetition_penalty=0.2,  # strictly positive float
        length_penalty=1.0,
        early_stopping=True,
        num_return_sequences=2,
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
            clean_up_tokenization_spaces=False,
        )


# top_p is
