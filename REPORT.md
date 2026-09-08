# Augmented planning decision-making: approach summary

## Method

The tool models an officer's workflow as a staged pipeline (`src/pipeline.py`); each
stage is a small, independently testable module.

1. **Ingest (`src/ingest/case_loader.py`).** The application pack is parsed with the
   provided kreuzberg parser, then cleaned. The packs are deliberately messy: every
   document appears twice (hyphenated and underscored) and some ship superseded or
   refused variants. Files are grouped by a normalized stem with any
   superseded/refused marker stripped, and one survivor is chosen per group — always
   preferring a live version over a superseded one. Every dropped file is recorded for
   the audit trail. Surviving documents are classified (application form, design &
   access statement, heritage/flood statements, elevations, site/floor plans,
   consultation replies), which drives both the vision pass and later weighting of
   consultee objections. Only drawings are parsed with image extraction, to save work.

2. **Site constraints (`src/tools/geospatial.py`, provided).** `postcode_lookup`
   returns deterministic constraints — flood zone, conservation area, green belt,
   listed-building grade, heritage-at-risk. It is also exposed to the assessment agent
   as a callable tool. Unknown postcodes are treated as "cannot confirm", never as
   "unconstrained".

3. **Case profile, incl. vision (`src/assessment/profile.py`).** A text pass over the
   form, statements, and consultation replies extracts a structured `CaseProfile`
   (proposal, address/postcode, application type, consultee positions, drawing
   references). A **vision pass** sends the drawing images to a vision model to recover
   ridge/eaves heights, storeys, separation distances, parking, and title-block drawing
   references that the text usually omits. The two are merged.

4. **Policy retrieval — embeddings RAG (`src/policy/`).** The ~100 MB policy corpus
   (NPPF, legislation, Doncaster Local Plan, SPDs, conservation-area appraisals) is far
   too large to place in a prompt and the brief describes officers making "multiple
   lookups", so this is a retrieval problem. Every policy PDF is parsed to text once,
   chunked into overlapping windows tagged with any detectable policy reference (e.g.
   "Policy 44", "NPPF paragraph 135"), embedded with a **local** model
   (`fastembed`, `BAAI/bge-small-en-v1.5`), and cached to disk as a numpy matrix plus a
   JSONL sidecar. Embeddings are local because OpenRouter is a chat gateway and does not
   reliably expose an embeddings endpoint; this also makes retrieval free and offline.
   Retrieval is cosine similarity over the normalized matrix.

5. **Assessment agent (`src/assessment/agent.py`).** A tool-calling loop over the chat
   completions API is given the profile and constraints plus two tools — `policy_search`
   and `postcode_lookup` — and works through a fixed checklist of material
   considerations (principle, design, heritage, amenity, highways/parking,
   flood/drainage, ecology/trees, contamination). It looks policy up per theme and
   produces, for each theme, a decision-notice-style finding, the policies cited, a harm
   level (`none` / `conditionable` / `unresolved`), and — where conditionable — a
   suggested condition.

6. **Synthesis (`src/assessment/synthesize.py`).** The decision boundary is
   deterministic and auditable: **refuse if any theme carries unresolved harm, otherwise
   approve with conditions.** Reasons are the per-theme findings in canonical order
   (acceptable themes included, matching the ground-truth style). Approvals prepend the
   two standard conditions — the Section 91 three-year commencement limit and the
   "approved plans" condition built from the profile's drawing references — followed by
   each theme's suggested condition, each ending with its `Reason:` clause. Output is the
   provided `Decision` model, giving both the required JSON and an officer-facing
   markdown draft.

**Why this shape.** Retrieval is forced by the corpus size and matches the brief's
multi-lookup workflow; the per-theme structure mirrors how the ground-truth decisions are
written and makes the reasons list align with the evaluator's semantic matching;
determinism at the decision boundary keeps the outcome transparent and reproducible,
which the brief calls out as essential.

**Models are configurable.** Chat, vision, and embedding models are all set via
environment variables (`PLANNING_CHAT_MODEL`, `PLANNING_VISION_MODEL`,
`PLANNING_EMBED_MODEL`) with cheap-but-capable defaults, so a stronger model can be
swapped in without code changes.

## How to run

```bash
uv sync                                  # install dependencies (adds fastembed)
uv run assignment-setup                  # decrypt OpenRouter creds into .env
uv run planning build-index              # one-time policy embedding index (cached)
uv run python scripts/evaluate_dev.py    # score against training cases 001-005
uv run planning batch                    # generate predictions for holdout 006-010
uv run python scripts/validate_submission.py
```

## Error analysis

*To be completed against measured dev-set results (cases 001–005) using
`scripts/evaluate_dev.py`, which reports `decision_match` and similarity-weighted
reasoning precision/recall/F1 per case and in aggregate.*

Anticipated failure modes to examine:
- **Over-refusal:** classifying a conditionable harm as `unresolved`. Mitigated by the
  explicit three-level harm taxonomy in the agent prompt.
- **Citation drift:** citing a plausible but wrong policy number. Retrieval tags chunks
  with detected references to keep citations grounded.
- **Missed themes:** a material consideration not raised at all, hurting recall. Mitigated
  by the fixed theme checklist.
- **Drawing-reference gaps:** weak title-block OCR degrading the "approved plans"
  condition.

## Future work

- Add a lightweight LLM "polish" pass over the deterministic reasons if measured F1
  warrants it, while keeping the decision boundary deterministic.
- Cross-check cited policy numbers against the retrieved chunks and flag unsupported
  citations.
- Cache per-case profiles to make prompt/retrieval iteration cheaper.
- Broaden `postcode_lookup` handling of unknown sites with a genuine geospatial fallback.

## Use of coding assistants

This solution was built with the assistance of v0. The architecture, module boundaries,
decision logic (derived from analysing the five worked cases), and prompt design were
directed and reviewed by me; the assistant produced the implementation, which was then
verified with ruff, mypy, and unit checks of the deterministic ingestion and synthesis
logic.
