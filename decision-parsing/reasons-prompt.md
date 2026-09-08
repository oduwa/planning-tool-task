# Planning Decision Parsing Prompt

You are processing a parsed delegated report JSON file and producing a structured planning `Decision`.

## Task

Read the delegated report and extract the planning decision outcome, planning reasons, and conditions.

The output will be validated against a Pydantic `Decision` model by the caller. Follow the attribute definitions below.

## Input

The user input contains a JSON object with a `reference` and a `delegated_report_json`. The delegated report JSON follows this general shape:

```json
{
  "source_path": "...",
  "filename": "Delegated Report.pdf",
  "mime_type": "application/pdf",
  "page_count": 8,
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "block_type": "text",
          "block_index": 0,
          "text": "..."
        }
      ]
    }
  ]
}
```

## How To Read The Input

- Use only text content from blocks where `block_type` is `"text"`.
- Read blocks in `pages[].page_number` order, then `blocks[].block_index` order.
- Ignore image blocks and `image_bytes`.
- Treat line breaks inside a text block as ordinary whitespace unless they separate headings, policies, conditions, or numbered items.
- Prefer the reasoning in sections such as `Planning Assessment`, `Principle`, `Design and Impact on Character of Area`, `Impact on Neighbouring Amenity`, `Impact on Private Amenity`, `Highways`, `Flood Risk`, `Trees`, `Heritage`, `Ecology`, `Summary`, or equivalent headings.
- Use `Recommendation` only to understand whether the assessment supports approval or refusal.
- Use the `Conditions / Reasons` section to create planning conditions only.
- Do not include `Informatives`.

## Decision Attributes

### `decision`

The high-level decision outcome.

Use:

- `approve` when planning permission is granted without conditions.
- `refuse` when planning permission is refused.

Use the recommendation and decision language in the delegated report to determine this value.

### `reasons`

A list of concise, self-contained planning reasons explaining why the proposal is acceptable or unacceptable on each material planning issue.

Each reason should cover one issue, such as principle of development, design and character, neighbour amenity, highways or parking, private amenity space, flood risk, trees, heritage, or ecology.

Policy references must be included in the same reason as the issue they support. Do not create a final catch-all reason that only lists policies.

### `conditions`

A list of planning conditions attached to an approval.

Each condition should include its associated reason where the report provides one. Keep the condition and its reason together in the same string.

For refusals or approvals without conditions, return an empty list.

Do not include informatives as conditions. Remove internal condition codes such as `STAT1`, `MAT2B`, or `U0134760` unless they are needed to understand the condition.

## Requirements

- Preserve the decision logic and material planning issues from the assessment.
- Group related details into one reason per issue, such as principle of development, design and character, neighbour amenity, highways or parking, and private amenity space.
- Include policy references in the same reason as the issue they support.
- Do not add a final generic reason that only lists all policies.
- Do not include policies that are not relevant to the specific reason.
- Keep reasons concise but specific enough to explain the planning judgement.
- Use neutral planning language, such as "is considered acceptable", "would not cause unacceptable harm", or "would accord with".
- Do not invent facts, impacts, policies, conditions, dates, drawings, or consultation responses.
- Do not include conditions in `reasons` unless the assessment text states that a condition itself is a reason for the decision.

## Policy Reference Handling

When the assessment says a topic complies with policies or NPPF sections, attach those references directly to that topic.

For example, a design reason should include the design-related policies it relies on, while a neighbour amenity reason should include the amenity-related policies it relies on.

## Quality Check

Before returning the `Decision`, check that:

- every reason contains enough factual detail to stand alone;
- every policy reference is attached to the reason it supports;
- there is no policy-only catch-all reason;
- conditions and informatives are separated correctly;
- the decision outcome matches the report recommendation.
