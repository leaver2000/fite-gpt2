__all__ = ["pipeline"]
from transformers import TextGenerationPipeline, GPT2LMHeadModel

from .gpt2 import (
    MODEL_PATH,
    MAX_LENGTH,
    tokenizer,
    device,
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"The model must be trained before it can be used. Run `python -m model.gpt2` to train the model."
    )

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).cuda(device)  # type: ignore
pipe = TextGenerationPipeline(
    model=model,  # PreTrainedModel
    tokenizer=tokenizer,  # PreTrainedTokenizer
    device=device,  # torch.device
    framework="pt",  # Literal["pt", "tf"]
    temperature=0.2,  # strictly positive float
    top_k=100,  # int
    top_p=0.1,  # float
    # repetition_penalty=2.0,  # float
    # do_sample=True,  # bool
)


def pipeline(text: str, max_length: int = MAX_LENGTH):
    return pipe(text, max_length=max_length)
