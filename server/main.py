from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import pipeline


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

@app.get("/")
async def root():
    """health check"""
    return {"message": "Hello World"}


@app.post("/generate/")
def generate(request:Request):
    """post route to generate a terminal aerodrome forecast"""
    print("Generating TAF for: ", request.userInput)
    results = pipeline(request.userInput.upper())  # type: ignore
    print("Results: ", results)
    return [{"generated_text":result["generated_text"].upper()} for result in results] # type: ignore