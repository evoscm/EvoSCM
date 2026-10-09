def complete_response(
    model,
    messages,
    system,
    max_tokens,
    api_key,
    base_url,
    timeout,
    max_retries,
    headers=None,
):
    from openai import OpenAI

    kwargs = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": min(2, max_retries),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["default_headers"] = headers
    client = OpenAI(**kwargs)
    response = client.responses.create(
        model=model,
        instructions=system,
        input=messages,
        max_output_tokens=max_tokens,
        store=False,
    )
    text = response.output_text
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Responses API returned no output_text")
    return text
