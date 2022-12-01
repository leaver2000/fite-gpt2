from datetime import datetime
from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .enum import EnumBase, StrEnum
from .pipeline import CodePredictionPipeline, Strategys
from .util import FileSystemDirectory

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
fsd = FileSystemDirectory()


class PipelineEngine(EnumBase):
    """Pipeline engine to use for code generation."""

    value: CodePredictionPipeline
    taf = fsd.get_pipeline("taf")

    @classmethod
    def get_pipeline(cls, pipeline: Union[StrEnum, str]) -> CodePredictionPipeline:
        return cls[pipeline].value

    @classmethod
    def generate(
        cls,
        pipeline: Union[StrEnum, str],
        text: str | list[str],
        strategy: Strategys | str | None = None,
    ) -> list[list[str]]:
        return cls.get_pipeline(pipeline).generate(text, strategy=strategy)


Pipelines: type[StrEnum] = StrEnum(
    "Pipelines", PipelineEngine.list_members()
)  # type: ignore


class Prompt(BaseModel):
    text: str | list[str] = [
        "TAF KBLV 010000Z 0103/0112 00000KT 9999 SCT250",
        "TAF KDAA 010000Z 0103/0112 00000KT 5000",
    ]


@app.get("/")
def read_root():
    return {"": datetime.now()}


@app.get("/models")
def list_model():
    return fsd.list_models()


@app.get("/models/{name}")
def get_model(name: str):
    return fsd.get(name)


@app.get("/strategys")
def list_strategys():
    return Strategys.list_members()


@app.get("/strategys/{name}")
def get_strategys(name: str):
    return Strategys[name].value


@app.post("/generate/{pipeline}")
def generate(prompt: Prompt, pipeline: Pipelines, strategy: Strategys):  # type: ignore
    return PipelineEngine.generate(pipeline, prompt.text, strategy=strategy)
