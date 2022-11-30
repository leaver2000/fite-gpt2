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

__all__ = [
    "Engine",
]
VERSION = "0.1.0"


def split_version(version: str):
    """Map version to major, minor, patch"""
    return tuple(map(int, version.split(".")))


def get_latests_version(base_model: str, dataset_name: str):
    """Get the latest version of the model"""
    store = Path.cwd() / "store"
    fs = {
        split_version(folder.name.split("-")[-1]): folder
        for folder in store.glob(f"{base_model}-{dataset_name}-*")
    }
    return fs[max(fs.keys())]


TAF_PATH = get_latests_version("gpt2", "taf")


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
    """Collection of pipelines for different models"""

    TAF: CodePredictionPipeline

    def __getitem__(self, key: Any) -> CodePredictionPipeline:
        return getattr(self, key)


PipelineEngineStrategys: TypeAlias = StrEnum(
    "PipelineEngineStrategys", PipelineEngine._fields
)


Engine = PipelineEngine(
    TAF=CodePredictionPipeline(
        model=GPT2LMHeadModel.from_pretrained(TAF_PATH / "model"),  # type: ignore
        tokenizer=GPT2TokenizerFast.from_pretrained(TAF_PATH / "tokenizer"),  # type: ignore
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
