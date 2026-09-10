"""OpenRouter (OpenAI-compatible) client, model config, and JSON helpers.

The assignment provisions an OpenRouter key via ``assignment-setup`` and stores
``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` in ``.env``. OpenRouter exposes
the OpenAI **chat completions** API (not the Responses API), so everything here
uses ``client.chat.completions`` for portability across models.
"""

from __future__ import annotations

import json
import os
import re
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
    """Per-component model selection and sampling controls for the pipeline.

    Model choice is deliberately made *per component* rather than once for the
    whole system, because the components have different difficulty:

    - ``profile_model``: structured fact extraction from mostly clean text — an
      easy task where a small, cheap model is appropriate.
    - ``vision_model``: reading rasterised CAD drawings (dimensions, layout) — a
      genuinely multimodal task that needs a capable vision model.
    - ``reasoning_model``: the material-consideration judgment and refusal logic —
      the hard, high-stakes step that warrants the strongest available model.
    - ``embed_model``: a local sentence-embedding model for policy retrieval; runs
      offline, so there is no per-query cost and no reason to call a hosted model.

    Every value is overridable via ``PLANNING_*`` environment variables so a
    stronger (or cheaper) model can be swapped in per component without code
    changes. Determinism controls (``seed`` and an optional OpenRouter
    ``provider_order`` pin) reduce, though on a hosted API cannot fully eliminate,
    run-to-run variance. They are automatically omitted for o-series reasoning
    models (e.g. ``openai/o3``), which reject a custom ``temperature`` and ignore
    ``seed``; those models manage their own internal sampling instead.
    """

    profile_model: str = "openai/gpt-4o-mini"
    reasoning_model: str = "openai/o3"
    vision_model: str = "openai/gpt-4o-mini"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    temperature: float = 0.0
    seed: int | None = 7
    enable_vision: bool = True
    max_vision_pages: int = 6
    provider_order: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Build a config, applying ``PLANNING_*`` environment overrides."""
        provider_raw = os.getenv("PLANNING_PROVIDER_ORDER", "")
        provider_order = tuple(
            part.strip() for part in provider_raw.split(",") if part.strip()
        )
        seed_raw = os.getenv("PLANNING_SEED", str(cls.seed))
        seed = None if seed_raw.lower() in {"", "none"} else int(seed_raw)
        return cls(
            profile_model=os.getenv("PLANNING_PROFILE_MODEL", cls.profile_model),
            reasoning_model=os.getenv("PLANNING_REASONING_MODEL", cls.reasoning_model),
            vision_model=os.getenv("PLANNING_VISION_MODEL", cls.vision_model),
            embed_model=os.getenv("PLANNING_EMBED_MODEL", cls.embed_model),
            temperature=float(os.getenv("PLANNING_TEMPERATURE", str(cls.temperature))),
            seed=seed,
            enable_vision=os.getenv("PLANNING_ENABLE_VISION", "1").lower()
            not in {"0", "false", "no"},
            max_vision_pages=int(
                os.getenv("PLANNING_MAX_VISION_PAGES", str(cls.max_vision_pages))
            ),
            provider_order=provider_order,
        )

    def provider_extra_body(self) -> dict[str, Any] | None:
        """Return an OpenRouter ``extra_body`` provider pin, if one is configured.

        Pinning the backend provider (and disabling fallbacks) is the strongest
        reproducibility lever available on OpenRouter: it keeps every request on
        the same hardware/build so greedy decoding stays stable.
        """
        if not self.provider_order:
            return None
        return {
            "provider": {
                "order": list(self.provider_order),
                "allow_fallbacks": False,
            }
        }


def is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI o-series reasoning models (o1/o3/o4...).

    These models reject a non-default ``temperature`` and do not honour ``seed``,
    so those sampling parameters must be omitted from the request.

    Args:
        model: Model id, optionally provider-prefixed (e.g. ``openai/o3``).
    """
    name = model.split("/")[-1]
    return re.match(r"o\d", name) is not None


def sampling_params(
    model: str, temperature: float, seed: int | None
) -> dict[str, Any]:
    """Return the sampling kwargs a chat request should send for ``model``.

    For reasoning models this is empty (let the API apply its own defaults); for
    standard models it carries the configured ``temperature`` and ``seed``.

    Args:
        model: Model id to call.
        temperature: Desired sampling temperature.
        seed: Desired best-effort reproducibility seed, if any.
    """
    if is_reasoning_model(model):
        return {}
    return {"temperature": temperature, "seed": seed}


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
    seed: int | None = None,
    extra_body: dict[str, Any] | None = None,
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
            response_format={"type": "json_object"},
            extra_body=extra_body,
            **sampling_params(model, temperature, seed),
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
