# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster
USER root
WORKDIR /
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY store/models/gpt2-taf-0.1.0 /opt/model
COPY store/tokenizer/gpt2-taf-0.1.0 /opt/tokenizer

ENV FITE_MODEL_PATH="/opt/model"
ENV FITE_TOKENIZER_PATH="/opt/tokenizer"

WORKDIR /code
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app/ /app
# run the uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]



