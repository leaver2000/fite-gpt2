from datetime import datetime
from typing import Any, TypeAlias

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ._enum import StrEnum
from .pipeline import PipelineEngine, Strategies

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


engine = PipelineEngine.load_from_pyproject("pyproject.toml")

_Pipelines: type[StrEnum] = StrEnum("Pipelines", engine.list_models())  # type: ignore
Pipelines: TypeAlias = _Pipelines


class Prompt(BaseModel):
    metadata: str | None = None
    text: str = "TAF KBLV 201853Z 2019/2118 00000KT P6SM SCT250"


@app.get("/")
def read_root() -> dict[str, str]:
    return {"": datetime.now().isoformat()}


@app.get("/models")
def list_model() -> list[str]:
    return engine.list_models()


@app.get("/models/{name}")
def get_model(name: str) -> dict[str, Any]:
    return engine.get_model(name).config.to_dict()


@app.get("/strategies")
def list_strategies() -> list[str]:
    return Strategies.list_members()


@app.get("/strategies/{name}")
def get_strategies(name: str) -> str:
    return Strategies[name].value


@app.post("/generate", response_model=list[list[str]] | list[str])
def generate(
    prompt: Prompt | list[Prompt], model: Pipelines, strategy: Strategies
) -> list[list[str]] | list[str]:
    if isinstance(prompt, list):
        text = [p.text for p in prompt]
    else:
        text = prompt.text
    return engine.generate(model, text, strategy=strategy)
