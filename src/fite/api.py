import re
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


class Metadata(BaseModel):
    TX: float | None = None
    TN: float | None = None
    issueDatetime: datetime = datetime.now()

class Prompt(BaseModel):
    metadata: Metadata = Metadata(TX=0,TN=0,issueDatetime=datetime.now())
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


def format_metadata(md:Metadata) -> str:
    def make_temp(prefix:str, temp:float | None) -> str:
        if temp is  None:
            return "<|unk|>"
        else:
            return f'{prefix}{int(temp):02}'.replace("-","M")
    tx_ = make_temp("TX", md.TX)
    tn_ = make_temp("TN", md.TN)

    return f"TAF [{tx_} {tn_} {md.issueDatetime:%Y-%m-%d}]"



@app.post("/generate", response_model=list[str])
def generate(
    prompt: Prompt, model: Pipelines, strategy: Strategies
) ->  list[str]:
    """
    Each Prompt.text should begin with either the 4-letter ICAO or AMD ICAO
    
    Example:
    >>> {
        "metadata": {
            "TX": 0,
            "TN": 0,
            "issueDatetime": "2021-09-20T18:53:00"
        },
        "text": "KBLV 201853Z 2019/2118 00000KT P6SM SCT250"
    }
    """
    # adding the metadata to the first line of the forecast
    text = (format_metadata(prompt.metadata) +" "+ prompt.text.lstrip("TAF").strip())
    # generating the forecast
    first, *rest = engine.generate(model, text, strategy=strategy)
    # removing the metadata from the first line of the forecast
    first = re.sub(r"(?<=^TAF)\s+\[.*\]\s+"," ",first)
    return [first, *rest]
