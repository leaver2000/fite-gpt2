from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import gpt2


class Request(BaseModel):
    userInput: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
taf_pipeline = gpt2.pipeline()
PIPELINES = {
    "TAF": taf_pipeline,
}


@app.get("/")
async def root():
    """health check"""
    return {"message": "Hello World"}


@app.post("/generate/{model_name}")
def generate(request: Request, model_name: Literal["TAF"]):
    """post route to generate a terminal aerodrome forecast"""
    print(f"Generating {model_name} for: ", request.userInput)

    results = PIPELINES[model_name].generate_forecast(request.userInput.upper())
    print("Results: ", results)
    return [{"generated_text": "\n ".join(result)} for result in results]  # type: ignore
