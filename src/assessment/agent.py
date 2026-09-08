"""Theme-by-theme assessment via a tool-calling loop.

The agent is given the case profile and site constraints, plus two tools:
``policy_search`` (semantic retrieval over the policy index) and
``postcode_lookup`` (deterministic site constraints). It makes as many policy
lookups as it needs — mirroring how an officer consults multiple policies — then
emits a structured, per-theme assessment with citations.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from assessment.schema import THEME_ORDER, CaseAssessment, CaseProfile
from llm import ModelConfig, complete_json
from policy.retriever import PolicyRetriever
from tools.geospatial import postcode_lookup

MAX_TOOL_ITERATIONS = 14
SEARCH_K = 6

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "policy_search",
            "description": (
                "Semantic search over the Doncaster Local Plan, SPDs, and national "
                "policy (NPPF, legislation). Use it to find the specific policies and "
                "paragraphs that apply to a planning theme, so findings can cite them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy question, e.g. 'parking standards for a new dwelling'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "postcode_lookup",
            "description": (
                "Return geospatial planning constraints for a site postcode: flood "
                "zone, conservation area, green belt, listed building grade, heritage "
                "at risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "postcode": {"type": "string", "description": "Site postcode."}
                },
                "required": ["postcode"],
            },
        },
    },
]

_SYSTEM = (
    "You are a UK planning officer for Doncaster Council assessing a housing "
    "application. Work through each material consideration in turn: principle of "
    "development, design and character, heritage, residential amenity, highways and "
    "parking, flood risk and drainage, ecology and trees, and contamination and "
    "environmental health.\n\n"
    "For every theme, first use the policy_search tool to find the exact Local Plan "
    "policies and NPPF paragraphs that apply, and use postcode_lookup to confirm site "
    "constraints. Then reach a reasoned judgment. Classify each theme's harm as:\n"
    "- 'none' when the theme is acceptable;\n"
    "- 'conditionable' when acceptable subject to a planning condition (provide the "
    "condition, ending with a 'Reason:' clause);\n"
    "- 'unresolved' when there is material harm that cannot be conditioned away and "
    "which would justify refusal;\n"
    "- 'not_applicable' when the theme genuinely does not arise.\n\n"
    "An application is refused if any theme is 'unresolved', otherwise approved with "
    "conditions. Write each finding as a self-contained decision-notice paragraph that "
    "names the policies relied on (e.g. 'in accordance with Doncaster Local Plan "
    "Policies 13 and 44' or 'contrary to NPPF paragraph 135'). Make multiple policy "
    "searches as needed before concluding."
)


def _dispatch_tool(
    name: str, arguments: dict[str, Any], retriever: PolicyRetriever
) -> str:
    """Execute a tool call and return a string result for the model.

    Args:
        name: Tool name.
        arguments: Parsed tool arguments.
        retriever: Loaded policy retriever.
    """
    if name == "policy_search":
        query = str(arguments.get("query", "")).strip()
        chunks = retriever.search(query, k=SEARCH_K)
        return PolicyRetriever.format_results(chunks)
    if name == "postcode_lookup":
        return json.dumps(postcode_lookup(str(arguments.get("postcode", ""))))
    return f"Unknown tool: {name}"


def _profile_brief(profile: CaseProfile) -> str:
    """Render the profile as a compact briefing for the agent.

    Args:
        profile: The extracted case profile.
    """
    consultees = "\n".join(
        f"  - {c.body}: {c.stance} — {c.summary}" for c in profile.consultee_positions
    )
    dimensions = "\n".join(f"  - {d}" for d in profile.key_dimensions)
    return (
        f"Proposal: {profile.proposal_description}\n"
        f"Site: {profile.site_address}\n"
        f"Postcode: {profile.postcode or 'not stated'}\n"
        f"Application type: {profile.application_type}\n"
        f"Key dimensions:\n{dimensions or '  - none recorded'}\n"
        f"Consultee positions:\n{consultees or '  - none recorded'}\n"
        f"Other notes: {profile.raw_notes or 'none'}"
    )


async def assess_case(
    profile: CaseProfile,
    client: AsyncOpenAI,
    cfg: ModelConfig,
    retriever: PolicyRetriever,
) -> CaseAssessment:
    """Run the tool-calling assessment loop and return per-theme judgments.

    Args:
        profile: The extracted case profile.
        client: OpenAI-compatible async client.
        cfg: Model configuration.
        retriever: Loaded policy retriever.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Assess this application. Research policy with the tools, then you will "
                "be asked to produce the structured assessment.\n\n"
                f"{_profile_brief(profile)}"
            ),
        },
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = await client.chat.completions.create(  # type: ignore[call-overload]
            model=cfg.chat_model,
            messages=messages,
            temperature=cfg.temperature,
            tools=_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = _dispatch_tool(call.function.name, arguments, retriever)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
        logger.info(
            f"Assessment iteration {iteration + 1}: handled {len(tool_calls)} tool call(s)"
        )

    themes_list = ", ".join(theme.value for theme in THEME_ORDER)
    final = await complete_json(
        client,
        cfg.chat_model,
        CaseAssessment,
        system=_SYSTEM,
        user=(
            "Using everything researched above, output the structured assessment now. "
            f"Include one entry for each applicable theme ({themes_list}). Omit a theme "
            "only if it genuinely does not arise. Each finding must cite the specific "
            "policies relied on.\n\n"
            f"Case briefing again for reference:\n{_profile_brief(profile)}\n\n"
            f"Research transcript:\n{_transcript(messages)}"
        ),
        temperature=cfg.temperature,
    )
    return final


def _transcript(messages: list[dict[str, Any]]) -> str:
    """Flatten the tool-call transcript into text for the final synthesis call.

    Args:
        messages: The running chat message list.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                lines.append(
                    f"[search] {call['function']['name']}({call['function']['arguments']})"
                )
        elif role == "tool":
            content = str(message.get("content", ""))
            lines.append(f"[result] {content[:1200]}")
        elif role == "assistant" and message.get("content"):
            lines.append(f"[notes] {message['content']}")
    return "\n".join(lines)
