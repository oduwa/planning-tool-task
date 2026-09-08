"""OpenRouter (OpenAI-compatible) client, model config, and JSON helpers.

The assignment provisions an OpenRouter key via ``assignment-setup`` and stores
``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` in ``.env``. OpenRouter exposes
the OpenAI **chat completions** API (not the Responses API), so everything here
uses ``client.chat.completions`` for portability across models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

OPENROUTER_API_KEY_NAME = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL_NAME = "OPENROUTER_BASE_URL"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelConfig:
    """Configurable model selection for the pipeline.

    All values are overridable via environment variables so the models can be
    swapped without code changes. Defaults are cheap-but-capable choices for
    development.
    """

    chat_model: str = "openai/gpt-4o-mini"
    vision_model: str = "openai/gpt-4o-mini"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    temperature: float = 0.0
    max_vision_images: int = 8

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Build a config, applying ``PLANNING_*`` environment overrides."""
        return cls(
            chat_model=os.getenv("PLANNING_CHAT_MODEL", cls.chat_model),
            vision_model=os.getenv("PLANNING_VISION_MODEL", cls.vision_model),
            embed_model=os.getenv("PLANNING_EMBED_MODEL", cls.embed_model),
            temperature=float(os.getenv("PLANNING_TEMPERATURE", str(cls.temperature))),
            max_vision_images=int(
                os.getenv("PLANNING_MAX_VISION_IMAGES", str(cls.max_vision_images))
            ),
        )


def get_client() -> AsyncOpenAI:
    """Return an ``AsyncOpenAI`` client pointed at OpenRouter.

    Raises:
        RuntimeError: If the API key is not present in the environment.
    """
    api_key = os.getenv(OPENROUTER_API_KEY_NAME)
    if not api_key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_NAME} is not set. Run `uv run assignment-setup` "
            "to populate .env with your OpenRouter credentials."
        )
    base_url = os.getenv(OPENROUTER_BASE_URL_NAME, DEFAULT_BASE_URL)
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def image_data_uri(image_bytes: bytes, image_format: str) -> str:
    """Encode raw image bytes as a base64 ``data:`` URI for vision messages.

    Args:
        image_bytes: Raw image bytes.
        image_format: Image format such as ``png`` or ``jpeg``.
    """
    import base64

    fmt = image_format.lower().lstrip(".")
    if fmt == "jpg":
        fmt = "jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{fmt};base64,{encoded}"


def _user_content(
    user: str, image_uris: list[str] | None
) -> str | list[dict[str, Any]]:
    """Build a chat user-message payload, attaching images when provided."""
    if not image_uris:
        return user
    parts: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for uri in image_uris:
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


async def complete_json[T: BaseModel](
    client: AsyncOpenAI,
    model: str,
    response_model: type[T],
    system: str,
    user: str,
    *,
    image_uris: list[str] | None = None,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> T:
    """Call chat completions and parse the reply into a Pydantic model.

    Uses JSON-object response formatting and validates against ``response_model``.
    On a validation or JSON error, retries with the error fed back to the model.

    Args:
        client: OpenAI-compatible async client.
        model: Model id to call.
        response_model: Pydantic model the reply must conform to.
        system: System instructions.
        user: User prompt.
        image_uris: Optional image data URIs for a vision pass.
        temperature: Sampling temperature.
        max_retries: Extra attempts after the first, with error feedback.

    Returns:
        A validated instance of ``response_model``.

    Raises:
        ValueError: If a valid response cannot be parsed within the retries.
    """
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    system_with_schema = (
        f"{system}\n\nRespond with a single JSON object that conforms exactly to "
        f"this JSON schema. Do not include markdown fences or commentary.\n\n"
        f"JSON schema:\n{schema}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_with_schema},
        {"role": "user", "content": _user_content(user, image_uris)},
    ]

    last_error = ""
    for attempt in range(max_retries + 1):
        response = await client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        try:
            return response_model.model_validate_json(_strip_fences(content))
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            last_error = str(error)
            logger.warning(
                f"JSON parse attempt {attempt + 1} failed for {response_model.__name__}: {last_error}"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was not valid. Fix these errors and return "
                        f"only the corrected JSON object:\n{last_error}"
                    ),
                }
            )

    raise ValueError(
        f"Could not obtain valid {response_model.__name__} JSON after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )


def _strip_fences(content: str) -> str:
    """Strip Markdown code fences a model may add around JSON."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -len("```")]
        # Remove a leading ``json`` language tag if present.
        if text.lstrip().startswith("json"):
            text = text.lstrip()[len("json") :]
    return text.strip()
