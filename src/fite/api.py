import enum
from datetime import datetime
from typing import TypeAlias, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipeline import CodePredictionPipeline, Strategys
from .util import FileSystemDirectory, StrEnum

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
fsd = FileSystemDirectory()


class PipelineEngine(enum.Enum):
    """Pipeline engine to use for code generation."""

    value: CodePredictionPipeline
    taf = fsd.get_pipeline("taf")

    @classmethod
    def get_pipeline(cls, pipeline: Union["Pipelines", str]) -> CodePredictionPipeline:
        if not isinstance(pipeline, str):
            pipeline = pipeline.name
        return cls[pipeline].value

    @classmethod
    def generate(
        cls,
        pipeline: Union["Pipelines", str],
        text: str | list[str],
        strategy: Strategys | str | None = None,
    ) -> list[list[str]]:
        return cls.get_pipeline(pipeline).generate(text, strategy=strategy)


Pipelines: TypeAlias = StrEnum(
    "Pipelines", PipelineEngine._member_names_, module=__name__
)


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


@app.post("/generate/{name}")
def generate(name: Pipelines, prompt: Prompt, strategy: Strategys):
    return PipelineEngine.generate(name, prompt.text, strategy=strategy)


# @app.post("/generate/{pipeline}/{strategy}")
# def generate(prompt: Prompt, pipeline: Pipelines, strategy: Strategys):
#     return PipelineEngine.generate(pipeline, prompt.text, strategy=strategy)
