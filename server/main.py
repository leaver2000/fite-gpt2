from fastapi import FastAPI
from pydantic import BaseModel

from model import pipeline


class Response(BaseModel):
    generated_text: str


app = FastAPI()


@app.get("/")
async def root():
    """health check"""
    return {"message": "Hello World"}


@app.post("/generate/", response_model=list[Response])
async def generate(text: str) -> list[Response]:
    """post route to generate a terminal aerodrome forecast"""
    return pipeline(text)  # type: ignore
