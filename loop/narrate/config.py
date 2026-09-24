"""Narration settings, read from the environment.

The template is the default. ARCHITECTURE.md §3.10 makes the LLM optional
("The demo never depends on the LLM") and the roadmap's Week 9 risk line makes
the fallback "the default path in CI", so Ollama is something a deployment
switches on, never something a missing variable silently depends on.

Ollama is the only LLM backend, by project decision: the narrator must be free
and self-hosted, so no Decision Card ever leaves the machine and no paid API
is a dependency. docs/RUNNING_THE_PROJECT.md §7 records the earlier
Gemini-only note this supersedes, and why.

Deliberately separate from `api.config.Settings` for the reason
`loop/monitor/database.py` gives: the monitor worker narrates, and it has no
business holding the API's JWT signing secret.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_TEMPLATE = "template"
BACKEND_OLLAMA = "ollama"
BACKENDS = (BACKEND_TEMPLATE, BACKEND_OLLAMA)


class NarrationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VITALLOOP_", extra="ignore")

    narration_backend: Literal["template", "ollama"] = BACKEND_TEMPLATE
    ollama_url: str = "http://localhost:11434"
    # §3.10 names Llama 3.1 8B; this is its Ollama tag.
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)

    def llm_source(self) -> str | None:
        """The label a narrative from the configured LLM carries."""
        if self.narration_backend == BACKEND_OLLAMA:
            return f"ollama/{self.ollama_model}"
        return None


def narration_settings() -> NarrationSettings:
    return NarrationSettings()
