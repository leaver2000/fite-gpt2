import os
import random
import datetime
from pathlib import Path
from typing import Optional, TypeAlias, NamedTuple, Any

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .util import StrEnum
from .pipeline import (
    CodePredictionPipeline,
    HyperParameterStrategy,
    HyperParameterStrategys,
)

VERSION = "0.1.0"

MODEL_PATH = Path(os.getenv("FITE_MODEL_PATH", "store/models"))
if not MODEL_PATH.exists():
    raise OSError(f"Model path {MODEL_PATH} does not exist")

TOKENIZER_PATH = Path(os.getenv("FITE_TOKENIZER_PATH", "store/tokenizer"))
if not TOKENIZER_PATH.exists():
    raise OSError(f"Tokenizer path {TOKENIZER_PATH} does not exist")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)


class Request(BaseModel):
    userInput: str = "TAF KBLV 181730Z 1818/1918 "


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineEngine(NamedTuple):
    TAF: CodePredictionPipeline

    def __getitem__(self, key: Any) -> CodePredictionPipeline:
        return getattr(self, key)


PipelineEngineStrategys: TypeAlias = StrEnum(
    "PipelineEngineStrategys", PipelineEngine._fields
)


Engine = PipelineEngine(
    TAF=CodePredictionPipeline(
        model=GPT2LMHeadModel.from_pretrained(MODEL_PATH / f"gpt2-taf-{VERSION}"),  # type: ignore
        tokenizer=GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH / f"gpt2-taf-{VERSION}"),  # type: ignore
        device=DEVICE,
        max_length=256,
        num_return_sequences=1,
    )
)


@app.get("/")
async def root():
    """health check"""
    return {"health": datetime.datetime.utcnow().isoformat()}


@app.post("/generate/{pipeline}/{strategy}", response_model=list[str])
def generate(
    request: Request,
    pipeline: PipelineEngineStrategys,
    strategy: Optional[HyperParameterStrategys] = None,
) -> list[str]:
    """generate code
    Args:
        request (Request): user input
        pipeline (Pipelines): pipeline name
        strategy (Optional[Strategys], optional): hyperparameter strategy. Defaults to None.
    Returns:
        list[str]: generated code
    Raises:
        ValueError: if pipeline is not found
    """
    if not strategy:
        strategy = random.choice(tuple(HyperParameterStrategy))

    (results,) = Engine[pipeline].generate_forecast(
        request.userInput.upper(),
        strategy=strategy,
    )
    return results
