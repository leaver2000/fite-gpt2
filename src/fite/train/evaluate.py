"""
this module is incomplete and is not used in the training process.
"""

import json
from pathlib import Path

import pandas as pd

from ..util import ResultRecord


def taf(
    prompt_text: str,
    generated_text: list[str],
) -> float:
    """Calculate the TAF score for a prompt and completion pair."""
    score = 0
    prompt = prompt_text.split("\n")[-1]
    if "TS" in prompt:
        # only one line should start with that prompt
        (last_prompt_line, *_) = (
            line for line in generated_text if line.startswith(prompt)
        )
        # if the prompt has a TS in it then the generated text should have a CB remark
        if "CB" in last_prompt_line:
            score += 1
        else:
            score -= 1
        # there should not be any lower case letters in the generated text
        if all(line.isupper() for line in generated_text):
            score += 1
        else:
            # any lowercase are a significant error
            score -= 5

    return 0.0


def validate():
    with open("results.json") as f:
        results = (ResultRecord(**result) for result in json.load(f))

    for result in results:
        score = 0.0
        # looking at the most last line in the prompt text
        prompt = result.prompt_text.split("\n")[-1]
        if "TS" in prompt:
            # only one line should start with that prompt
            (last_prompt_line,) = (
                line for line in result.generated_text if line.startswith(prompt)
            )
            # if the prompt has a TS in it then the generated text should have a CB remark
            if "CB" in last_prompt_line:
                score += 1
            else:
                score -= 1
            # there should not be any lower case letters in the generated text
            if all(line.isupper() for line in result.generated_text):
                score += 1
            else:
                score -= 1
        result.score = score
        yield result.to_dict()


def validation_check(result_path: Path) -> ...:
    df = (
        pd.DataFrame(validate())
        .drop(columns=["hyper_parameters", "model"])
        .set_index("strategy")
    )
    for name, prompt_text, generated_text, score in df.sort_values(
        by=["score"], ascending=True
    ).itertuples():
        generated_text = "\n ".join(generated_text)
        print(
            f"""
{name=} {prompt_text=} {score=}
{generated_text}"""
        )
