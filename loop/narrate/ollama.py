"""The self-hosted narration backend: Ollama, as ARCHITECTURE.md §3.10 specifies.

The model is asked for determinism it can offer (`temperature` 0, a fixed
`seed`) but nothing downstream relies on it: whatever comes back is grounding-
checked, and a failure falls back to the template. Every error -- connection
refused, timeout, a malformed body, an empty answer -- surfaces as
`OllamaUnavailable`, so the narrator has one thing to catch.

The prompt, and the argument that it cannot carry PHI, live in `prompt.py`.
"""

import httpx

from loop.narrate.prompt import (
    SYSTEM_PROMPT,
    LLMUnavailable,
    build_prompt,
    card_for_prompt,
)

__all__ = ["SYSTEM_PROMPT", "OllamaUnavailable", "build_prompt", "card_for_prompt", "generate"]


class OllamaUnavailable(LLMUnavailable):
    """Ollama could not produce a narrative."""


def generate(
    card: dict,
    *,
    url: str,
    model: str,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> str:
    """One non-streaming `/api/generate` call. Returns the text or raises."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": build_prompt(card),
        "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }
    owned = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.post(f"{url.rstrip('/')}/api/generate", json=payload)
        response.raise_for_status()
        text = response.json().get("response")
    except (httpx.HTTPError, ValueError, AttributeError) as error:
        raise OllamaUnavailable(f"{type(error).__name__}: {error}") from error
    finally:
        if owned:
            client.close()

    if not isinstance(text, str) or not text.strip():
        raise OllamaUnavailable("Ollama returned an empty narrative")
    return text.strip()
