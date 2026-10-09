import os

HTTP_TIMEOUT_S = 300.0
OPENAI_MAX_RETRIES = 8
OPENAI_API_TRANSPORT = "responses"
OPENAI_API_ENDPOINT_PATH = "/v1/responses"


def complete(model, messages, system=None, max_tokens=4096):
    if model.startswith("claude"):
        return _claude_complete(model, messages, system, max_tokens)
    transport = os.environ.get("EVOSCM_API_TRANSPORT", "responses")
    if transport == "responses":
        from scienceagent.responses_transport import complete_response

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY or OPENAI_API_KEY_POOL_FILE must be set"
            )
        return complete_response(
            model,
            messages,
            system,
            max_tokens,
            api_key,
            os.environ.get("OPENAI_BASE_URL") or None,
            HTTP_TIMEOUT_S,
            OPENAI_MAX_RETRIES,
        )
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=HTTP_TIMEOUT_S,
        max_retries=min(2, OPENAI_MAX_RETRIES),
    )
    if transport == "chat":
        content = ([{"role": "system", "content": system}] if system else []) + messages
        response = client.chat.completions.create(
            model=model, messages=content, max_tokens=max_tokens
        )
        text = response.choices[0].message.content
    else:
        raise ValueError("EVOSCM_API_TRANSPORT must be responses or chat")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("The model returned no text")
    return text


def _claude_complete(model, messages, system, max_tokens):
    from anthropic import Anthropic

    base_url = os.environ.get("CLAUDE_BASE_URL", "https://api.anthropic.com").rstrip(
        "/"
    )
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if system:
        payload["system"] = system
    with Anthropic(
        api_key=os.environ["CLAUDE_API_KEY"],
        base_url=base_url,
        timeout=HTTP_TIMEOUT_S,
        max_retries=min(2, OPENAI_MAX_RETRIES),
    ) as client:
        response = client.messages.create(**payload)
    text = "\n".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise RuntimeError("The model returned no text")
    return text
