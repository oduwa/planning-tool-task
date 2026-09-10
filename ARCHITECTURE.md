# Architecture proposal — augmented planning decision-making

> **Status: original pre-build design proposal.** `REPORT.md` is the authoritative
> description of the system as built. This document is kept for provenance — it shows
> the design reasoning before implementation. The build kept the staged pipeline and
> the deterministic decision boundary described here, and went further in four areas:
> retrieval became **iterative** (MMR diversity + a `policy_lookup` citation tool)
> rather than single-pass; the plan **vision pass rasterises the vector-CAD sheets
> with PyMuPDF** (they carry no embedded raster images, so `doc.get_images()` returns
> nothing); a substantive **guardrail layer** (`src/guardrails.py`) was added; and
> **model choice is per-component** with committed evaluation evidence under
> `evaluation/`.

---

## 1. What the data tells us (design constraints)

These observations are drawn directly from `data/doncaster/decisions/case-00{1..5}` and the
application packs, and they drive every design choice below.

### 1.1 Shape of a "good" decision

Each ground-truth `decision.json` has three fields: `decision`, `reasons`, `conditions`.
Consistent patterns:

- **`reasons` are per-theme planning judgments.** Each list item is a self-contained
  paragraph on one issue — *principle of development, design/character, heritage,
  residential amenity, highways/parking, flood risk/drainage, ecology, trees,
  contamination*. A case has ~5–9 reasons, one per material consideration.
- **Almost every reason cites specific policy.** e.g. *"...in accordance with Doncaster
  Local Plan Policies 1 and 23"*, *"...contrary to NPPF paragraph 135"*. Citation is
  explicitly required for "audit and transparency" (README).
- **Reasons are written even for themes that pass.** A refusal (case-001) still lists the
  themes that were *acceptable* alongside the ones that caused refusal. So the output is a
  full assessment, not just the deciding factor.
- **"as amended" / "amended plan" language recurs.** Officers assess the *latest* revision
  of drawings, which matters given the packs contain superseded versions (see 1.3).

### 1.2 Decision logic (approve vs refuse)

Derived from the five worked cases:

> **Refuse if any material theme has unresolved harm that cannot be conditioned away;
> otherwise approve with conditions.**

case-001 refuses even though 6 of 9 themes are acceptable, because refuse/recycling
access, noise mitigation, and cycle access were unresolved harms. The other four cases
approve because every harm was either absent or resolvable by condition.

### 1.3 Conditions (approvals only)

- Refusals have `conditions: []` (the validator enforces this).
- Approvals always open with **two standard conditions**:
  1. 3-year commencement limit — *"...Section 91 (as amended) of the Town and Country
     Planning Act 1990."*
  2. "in accordance with the approved plans" — **listing each drawing number + revision +
     received date**. This is why version tracking (1.4) is load-bearing, not cosmetic.
- Then **theme-specific conditions** (materials, drainage, ecology enhancement, tree
  protection, contamination, construction method). Every condition ends with a
  `Reason: ...` clause, usually citing a Local Plan policy.

### 1.4 The case packs are deliberately messy

`ls data/doncaster/applications/case-001` shows the trap:

- **Duplicate files** — every doc appears twice, once hyphenated
  (`application-form-no-personal-data.pdf`) and once with underscores
  (`application_form_no_personal_data.pdf`).
- **Superseded / refused variants** — `existing_and_proposed_elevations_superseded.pdf`,
  `design-and-access-statement-refused.pdf` sit alongside the live versions.

Feeding all of these to the model would double-count evidence and mix in withdrawn
proposals. **De-duplication and version selection are a required ingestion stage**, not an
afterthought.

### 1.5 Scale rules out context-stuffing

The policy corpus (`national_policy` + `local_policy` + `spds`) is ~100 MB of PDFs,
including 30 MB+ legislation. It cannot go in a prompt. The README also states officers
"make multiple lookups to different relevant policies" — this is a textbook **retrieval**
problem, matching the agreed embeddings-RAG direction.

---

## 2. Pipeline overview

The pipeline mirrors how an officer actually works: read the pack → establish site
constraints → identify the material issues → look up the relevant policy per issue →
weigh each issue → decide and draft.

```
                         ┌─────────────────────────────────────────┐
                         │  ONE-TIME: build policy index (offline)  │
                         │  parse policy PDFs → chunk → embed →      │
                         │  persist vectors  (src/policy/index.py)   │
                         └─────────────────────────────────────────┘
                                            │ (reused by every case)
   case-00X/ ─┐                             ▼
              │   1. INGEST            2. CONSTRAINTS        4. POLICY RAG
              ├──▶ load pack       ──▶ postcode_lookup() ─┐  semantic search
              │    dedupe + drop        (deterministic)   │  over policy index
              │    superseded                             │
              │        │                                  │
              │        ▼                                  ▼
              │   3. CASE PROFILE  ◀── vision pass    5. ASSESSMENT (agent)
              │   LLM extracts         over plans        theme-by-theme reasoning
              │   structured facts     (elevations,      with policy_search +
              │   from text            site, floor)      postcode_lookup tools
              │        │                                  │
              │        └──────────────┬───────────────────┘
              │                       ▼
              │                6. SYNTHESIS
              │                refuse if any unresolved harm else approve;
              │                compose cited reasons + (approval) conditions
              │                       │
              │                       ▼
              │                7. OUTPUT
              └──────────────▶ predictions/case-00X-prediction.json
                               + officer-facing markdown (Decision.to_markdown)
```

---

## 3. Stage-by-stage design

### Stage 1 — Ingest (`src/ingest/`)

**Goal:** turn a messy case folder into a clean, ordered set of documents.

- Use the existing async parser: `await extract_pdf_document(path)` → `PdfDocument`
  (`src/parser/parser.py`). It already extracts text blocks and images at 300 DPI.
- **De-duplicate** hyphen/underscore twins by normalizing the stem
  (`re.sub(r"[-_]+", "_", stem.lower())`) and keeping one. If both parse identically, keep
  either; if they differ, keep the larger text extraction.
- **Drop superseded/refused** documents by filename signal
  (`_superseded`, `-refused`, `_refused`, `-superseded`) unless it is the only version of
  that document. Record what was dropped for the audit trail.
- **Classify** each surviving document by type from its filename + first-page text:
  `application_form`, `design_access_statement`, `heritage_statement`,
  `flood_risk_assessment`, `elevations`, `site_plan`, `floor_plan`, `consultation_reply`,
  `other`. Classification decides which docs get the vision pass (Stage 3) and helps the
  assessment agent weight consultee objections.

```python
# src/ingest/case_loader.py
class CaseDocument(BaseModel):
    doc_type: DocType
    filename: str
    document: PdfDocument            # parsed, from existing parser
    is_plan: bool                    # elevations / site / floor plans → vision

class CasePack(BaseModel):
    case_id: str
    documents: list[CaseDocument]
    dropped: list[str]               # superseded/duplicate files, for the audit trail

async def load_case_pack(case_dir: Path) -> CasePack: ...
```

### Stage 2 — Site constraints (`src/tools/geospatial.py`, existing)

Deterministic and already provided. The pipeline calls `postcode_lookup(postcode)` once
the postcode is known (from Stage 3's profile) and threads the result — flood zone,
conservation area, green belt, listed building grade, heritage-at-risk — into the
assessment. It is also exposed to the agent as a callable tool so it can re-query.

Design note: `postcode_lookup` returns `found: False` for unknown postcodes rather than
"no constraints". The synthesis step must treat *unknown* as "cannot confirm constraints"
rather than "unconstrained", to avoid silently approving a flood-zone site.

### Stage 3 — Case profile, incl. vision on plans (`src/assessment/profile.py`)

**Goal:** a structured `CaseProfile` of the facts a decision hinges on.

- **Text pass:** one LLM call over the concatenated text of the form + statements +
  consultation replies (`doc.get_texts()`), extracting a typed profile.
- **Vision pass (agreed):** for each `is_plan` document, send the page images
  (`doc.get_images()` → base64 data URIs) to a vision-capable model to read dimensions,
  storey heights, separation distances, boundary treatments, and parking layout that the
  text usually omits. Only plan documents are sent, to control token cost.

```python
# src/assessment/profile.py
class CaseProfile(BaseModel):
    proposal_description: str
    site_address: str
    postcode: str | None
    application_type: str                 # householder, full, change of use, ...
    key_dimensions: list[str]             # e.g. "ridge height 8.2m", "3.1m to boundary"
    drawing_references: list[DrawingRef]  # number + revision + received date (for conditions)
    consultee_positions: list[ConsulteePosition]  # body, stance, summary
    raw_notes: str

async def build_case_profile(pack: CasePack, client: AsyncOpenAI, cfg: ModelConfig) -> CaseProfile: ...
```

`DrawingRef` is what lets Stage 6 generate the "approved plans" condition accurately.

### Stage 4 — Policy retrieval (`src/policy/`) — embeddings RAG

**One-time index build** (cached to disk, reused across all cases):

1. Parse every PDF under `data/national_policy`, `data/local_policy`, `data/local_policy/spds`.
2. Chunk by ~800-token windows with overlap, tagging each chunk with
   `{source_file, page, policy_ref?}`. Where a heading like "Policy 44" or "Paragraph 135"
   is detectable, capture it so retrieved chunks carry a citable reference.
3. Embed chunks and persist a simple on-disk index.

> **Open decision — embedding backend (see §6).** OpenRouter is an OpenAI-*chat*-compatible
> gateway and does not reliably expose an `/embeddings` endpoint. Recommendation: use a
> **local** embedding model (`fastembed`, `BAAI/bge-small-en-v1.5`) so retrieval has zero
> API cost and works offline. Vectors stored as `numpy` + a JSON sidecar; cosine similarity
> in-memory. No vector DB needed at this corpus size. This adds one dependency.

**Per-issue retrieval:** `policy_search(query, k)` returns the top-k chunks with their
`source_file` / `policy_ref`, exposed to the assessment agent as a tool so it can look up
policy for each theme independently — directly modelling "multiple lookups".

```python
# src/policy/index.py
class PolicyChunk(BaseModel):
    text: str
    source_file: str
    page: int
    policy_ref: str | None       # "Policy 44", "NPPF para 135", when detectable

def build_policy_index(data_dir: Path, index_dir: Path, cfg: ModelConfig) -> None: ...

# src/policy/retriever.py
class PolicyRetriever:
    def search(self, query: str, k: int = 6) -> list[PolicyChunk]: ...
```

### Stage 5 — Assessment agent (`src/assessment/agent.py`)

**Goal:** a per-theme judgment with citations. Uses the already-installed
`openai-agents` SDK for tool-calling.

- The agent is given the `CaseProfile`, the `LocationConstraints`, and two tools:
  `policy_search` (Stage 4) and `postcode_lookup` (Stage 2).
- It works through a fixed **checklist of material considerations** (principle, design,
  heritage, amenity, highways/parking, flood/drainage, ecology/trees, contamination),
  looking up policy per theme and producing a structured judgment. The fixed checklist
  makes coverage predictable and keeps reasons aligned with the ground-truth themes.

```python
# src/assessment/schema.py
class ThemeAssessment(BaseModel):
    theme: PlanningTheme
    finding: str                       # the reasoning paragraph, citation-ready
    harm: HarmLevel                    # none | conditionable | unresolved
    cited_policies: list[str]
    suggested_condition: str | None    # when harm == conditionable

class CaseAssessment(BaseModel):
    themes: list[ThemeAssessment]
```

`HarmLevel` is the hinge for Stage 6: `unresolved` → refuse.

### Stage 6 — Synthesis (`src/assessment/synthesize.py`)

Deterministic assembly (no new model call needed, or one final polish call):

- **Outcome:** `REFUSE` if any theme is `unresolved`, else `APPROVE`.
- **Reasons:** one paragraph per theme, in a canonical order, each carrying its citations —
  matching the ground-truth style (including stating acceptable themes).
- **Conditions (approve only):**
  - inject the two standard conditions (s91 time limit; approved-plans list built from
    `CaseProfile.drawing_references`),
  - then each theme's `suggested_condition`, each with its `Reason:` clause.
- Produces a `Decision` (the existing `src/decision.py` model) → free `to_markdown()` and
  JSON serialization.

### Stage 7 — Output & CLI (`src/main.py`)

Wire the existing Typer stub:

```bash
uv run python -m main generate-planning-assessment case-006      # single case
```

Writes `predictions/case-00X-prediction.json` (exact validator schema) and an officer-facing
`.md` draft. A small batch entrypoint runs all holdout cases.

---

## 4. Evaluation harness (`src/evaluator/`, extend)

The provided `PlanningDecisionSimpleEvaluator` scores `decision_match` + semantic
reasoning F1 via an LLM judge. Plan:

- Add `scripts/evaluate_dev.py` to run the full pipeline on `case-001..005`, compare against
  their `decision.json`, and print per-case + aggregate scores.
- Use this loop to tune the theme checklist, retrieval `k`, and prompts **before** touching
  the holdout cases — so prompt iteration is measured, not vibes.
- Holdout cases 006–010 are then run once to produce the submitted predictions.

---

## 5. Proposed module layout

Extends the boilerplate; nothing existing is deleted.

```
src/
  main.py                 # (edit) wire the Typer command → run pipeline
  pipeline.py             # (new) orchestrates stages 1–7 for one case
  ingest/
    case_loader.py        # (new) load pack, dedupe, drop superseded, classify
  policy/
    index.py              # (new) build + persist the policy embedding index
    retriever.py          # (new) cosine-similarity search → PolicyChunk[]
  assessment/
    profile.py            # text → CaseProfile
    vision.py             # rasterise plans (PyMuPDF) → PlanReadout, merge into profile
    agent.py              # iterative theme-by-theme assessment (search + lookup tools)
    schema.py             # ThemeAssessment / CaseAssessment / enums
    synthesize.py         # CaseAssessment + profile → Decision
  guardrails.py           # scope/grounding/consistency/approval-bias/abstention checks
  llm.py                  # AsyncOpenAI(OpenRouter) client + per-component ModelConfig
  parser/                 # (unchanged) existing PDF parser
  tools/                  # (unchanged) geospatial lookup
  evaluator/              # (unchanged) provided scorer, driven by scripts/evaluate_dev.py
  decision.py             # (unchanged) Decision output model
scripts/
  evaluate_dev.py         # score pipeline on cases 001–005; writes evaluation/runs/
  validate_submission.py  # (unchanged) provided validator
tests/                    # deterministic (no-API) unit tests
evaluation/               # committed evaluation evidence (traces, metrics, error analysis)
```

---

## 6. Open decisions for sign-off

1. **Embedding backend.** Recommend local `fastembed`/`bge-small` (offline, no API cost,
   robust) over trying OpenRouter `/embeddings` (unreliable). Adds one dependency. — *OK?*
2. **Models.** The restricted key has a small credit balance. Suggest a cheap-but-capable
   chat model for profiling/assessment and a cheap vision model, both configurable in
   `llm.py`. I'll confirm exact model IDs against live OpenRouter availability at build time.
   — *Any model preference / cost ceiling?*
3. **Final synthesis call.** Deterministic assembly is cheaper and more auditable; one
   optional LLM "polish" pass makes reasons read more naturally but costs tokens and adds
   variance. Recommend deterministic first, measure, add polish only if F1 needs it. — *OK?*
4. **Vision scope.** Send only classified plan documents (elevations/site/floor). If credit
   is tight we can cap to the single most relevant plan per type. — *OK?*

---

## 7. Draft `REPORT.md` — Method section (for review)

> This is the write-up I'd commit to `REPORT.md` once the build matches it.

**Method.** The tool models an officer's workflow as a staged pipeline. A case pack is first
ingested and cleaned — duplicate hyphen/underscore files are collapsed and superseded/refused
drawings are dropped — then each document is classified. Site constraints come from the
deterministic `postcode_lookup` tool (flood zone, conservation area, green belt, heritage). A
structured `CaseProfile` is extracted with an LLM, including a **vision pass over the plans and
elevations** to recover dimensions and separation distances the text omits. Relevant policy is
retrieved from an **embeddings index** built once over the national and local policy corpus,
exposed to a tool-calling **assessment agent** that reasons theme-by-theme (principle, design,
heritage, amenity, highways, flood risk, ecology, contamination), citing the specific Local
Plan policies and NPPF paragraphs behind each finding. A deterministic synthesis step applies
the decision rule — *refuse if any theme carries unresolved harm, otherwise approve with
conditions* — and drafts the reasons and, for approvals, the standard and theme-specific
conditions (each with its `Reason:` clause). Output conforms to the required `Decision` schema
and also renders an officer-facing markdown draft.

**Why this method.** Retrieval is forced by the ~100 MB policy corpus and matches the brief's
"multiple lookups" workflow; per-theme structure mirrors the ground-truth decisions and makes
the reasoning list align with how officers write; determinism at the decision boundary keeps
the outcome auditable, which the brief calls out as essential for transparency.

*(Error analysis, future work, and use-of-assistants sections to be completed against measured
dev-set results.)*
```

---

## 8. What I need from you

Sign-off on the four open decisions in §6 (or your preferences), after which I'll implement
in this order: `llm.py` → ingest → policy index → profile+vision → agent → synthesize →
wire CLI → dev-eval loop → tune → generate the five holdout predictions → finalize `REPORT.md`.
