def taf(
    prompt_text: str,
    generated_text: list[str],
) -> float:
    """Calculate the TAF score for a prompt and completion pair."""
    score = 0
    prompt = prompt_text.split("\n")[-1]
    if "TS" in prompt:
        # only one line should start with that prompt
        (last_prompt_line,) = (
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
