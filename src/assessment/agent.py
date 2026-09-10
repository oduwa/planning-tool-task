"""Theme-by-theme assessment via an iterative, tool-calling research loop.

The agent is given the case profile and site constraints, plus three tools:

- ``policy_search`` — diverse semantic retrieval over the whole policy corpus
  (Local Plan, SPDs, conservation-area appraisals, NPPF, legislation);
- ``policy_lookup`` — pull a *named* policy or paragraph in full, so a citation
  can be read before it is relied on;
- ``postcode_lookup`` — deterministic site constraints.

It is prompted to investigate iteratively — search, read, and if the evidence is
inconclusive refine the query or look a policy up directly — mirroring how an
officer makes multiple lookups across policies before concluding. Only once the
research is done does it emit the structured, per-theme assessment with citations.
The instructions deliberately guard against the approval bias seen in planning
LLMs: refusal is a first-class outcome and consultee objections must be engaged.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from assessment.schema import THEME_ORDER, CaseAssessment, CaseProfile
from llm import ModelConfig, complete_json, sampling_params
from policy.retriever import PolicyRetriever
from tools.geospatial import postcode_lookup

MAX_TOOL_ITERATIONS = 16
SEARCH_K = 6

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "policy_search",
            "description": (
                "Semantic search over the full Doncaster policy corpus (Local Plan, "
                "SPDs, conservation-area appraisals) and national policy (NPPF, "
                "legislation). Returns diverse passages. Use it to find the policies "
                "and paragraphs that apply to a theme. Search repeatedly with refined "
                "queries when the first results are inconclusive."
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
            "name": "policy_lookup",
            "description": (
                "Retrieve every passage that mentions a specific named policy or "
                "paragraph, e.g. 'Policy 44' or 'NPPF paragraph 135'. Use this to "
                "read a policy in full before citing it, and to confirm a policy "
                "actually exists before relying on it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "A citation such as 'Policy 13' or 'paragraph 135'.",
                    }
                },
                "required": ["reference"],
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
    "application on its planning merits. Work through each material consideration "
    "in turn: principle of development, design and character, heritage, residential "
    "amenity, highways and parking, flood risk and drainage, ecology and trees, and "
    "contamination and environmental health.\n\n"
    "Investigate like an officer, iteratively:\n"
    "1. For each theme, use policy_search to find the applicable Local Plan policies "
    "and NPPF paragraphs, and postcode_lookup to establish site constraints.\n"
    "2. Before you rely on a policy, use policy_lookup to read it in full and confirm "
    "it says what you think. Never cite a policy you have not seen in the tool "
    "results.\n"
    "3. If the evidence is inconclusive, do not guess — refine your query and search "
    "again, or look up a related policy. Keep digging until you can support a "
    "judgment with cited text.\n\n"
    "Classify each theme's harm as:\n"
    "- 'none' when the theme is genuinely acceptable;\n"
    "- 'conditionable' when acceptable only subject to a planning condition (state the "
    "condition, ending with a 'Reason:' clause);\n"
    "- 'unresolved' when there is material harm that cannot be conditioned away and "
    "which justifies refusal;\n"
    "- 'not_applicable' when the theme genuinely does not arise.\n\n"
    "Guard against approval bias. Approval is NOT the default. If a statutory or "
    "internal consultee objects, you must either resolve that objection with cited "
    "policy or record the theme as unresolved. A holding objection, an unmet "
    "standard, or a conflict with an adopted policy that cannot be conditioned away "
    "is a refusal — say so plainly. Do not soften material harm into a condition to "
    "avoid refusing. An application is refused if ANY theme is 'unresolved'; "
    "otherwise it is approved with conditions.\n\n"
    "Write each finding as a self-contained decision-notice paragraph that names the "
    "policies relied on (e.g. 'in accordance with Doncaster Local Plan Policies 13 "
    "and 44' or 'contrary to NPPF paragraph 135')."
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
        return PolicyRetriever.format_results(retriever.search(query, k=SEARCH_K))
    if name == "policy_lookup":
        reference = str(arguments.get("reference", "")).strip()
        chunks = retriever.lookup_reference(reference, k=SEARCH_K)
        if not chunks:
            return (
                f"No passage in the policy corpus mentions '{reference}'. Do not cite "
                "it; find the correct policy with policy_search."
            )
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
    """Run the iterative tool-calling assessment loop and return per-theme judgments.

    Args:
        profile: The extracted case profile.
        client: OpenAI-compatible async client.
        cfg: Model configuration.
        retriever: Loaded policy retriever.
    """
    extra_body = cfg.provider_extra_body()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Assess this application. Research the policy corpus with the tools "
                "until every material consideration is supported by cited policy "
                "text, then you will be asked to produce the structured assessment.\n\n"
                f"{_profile_brief(profile)}"
            ),
        },
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = await client.chat.completions.create(  # type: ignore[call-overload]
            model=cfg.reasoning_model,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            extra_body=extra_body,
            **sampling_params(cfg.reasoning_model, cfg.temperature, cfg.seed),
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
        cfg.reasoning_model,
        CaseAssessment,
        system=_SYSTEM,
        user=(
            "Using everything researched above, output the structured assessment now. "
            f"Include one entry for each applicable theme ({themes_list}). Omit a theme "
            "only if it genuinely does not arise. Each finding must cite the specific "
            "policies relied on, and only policies that appeared in the tool results.\n\n"
            f"Case briefing again for reference:\n{_profile_brief(profile)}\n\n"
            f"Research transcript:\n{_transcript(messages)}"
        ),
        temperature=cfg.temperature,
        seed=cfg.seed,
        extra_body=extra_body,
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
