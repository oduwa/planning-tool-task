# Augmented planning decision-making: method, evaluation, and limitations

## 1. Method

The tool models a planning officer's workflow as a staged pipeline
(`src/pipeline.py`); each stage is a small, independently testable module. A case
folder goes in; a `Decision` (the required JSON), an officer-facing markdown draft,
and a guardrail report come out.

```
ingest → profile (text) → plan vision pass → iterative policy assessment
       → deterministic synthesis → guardrails → Decision + draft + audit trail
```

### 1.1 Ingest and de-duplication (`src/ingest/case_loader.py`)

The packs are a bit messy: Some documents appear to be included twice with slight variants (hyphenated and
underscored) and some ship superseded/refused/withdrawn variants. To address this, files are grouped
by a normalised stem with any superseded marker stripped, and one survivor is chosen
per group — always preferring a live version over a superseded one. Every dropped
file is recorded for the audit trail. Survivors are classified (application form,
design & access, heritage/flood statements, elevations, site/floor plans,
consultation replies), which drives both the assembled text and the later weighting
of consultee objections.


### 1.2 Full-corpus policy retrieval (`src/policy/`)

The policy corpus is far too large for a prompt, and the brief describes officers making
"multiple lookups", so this is a good candidate for retrieval aided generation. **The index walks the entire
national and local policy trees** (`rglob("*.pdf")`), so coverage is the whole corpus,
not a hand-picked subset. Each PDF is chunked into overlapping windows tagged with any
detectable policy reference (e.g. "Policy 44", "NPPF paragraph 135"), embedded with a
**local** model (`BAAI/bge-small-en-v1.5`), and cached to disk as a numpy
matrix plus a JSONL sidecar — **persistent indexing** built once and reused.

Retrieval is not a single-pass top-k lookup:

- **`search`** re-ranks candidates with **Maximal Marginal Relevance** so repeated
  queries on a theme return *diverse* passages (different policies/documents) rather
  than near-duplicates — which is what makes iterative deepening add real evidence.
- **`lookup_reference`** pulls every passage that mentions a specific named policy or
  paragraph, so a citation can be read in full before it is relied on, and so a claimed
  policy can be checked for existence.

### 1.3 Multimodal plan reading (`src/assessment/vision.py`)

The documents also contain diagrams and images. However, the drawings appear to be
vector CAD images, I therefore
**rasterise each drawing page with PyMuPDF** (~144 dpi) and send the PNGs to a vision
model, which recovers ridge/eaves heights, storeys, separation distances, parking and
title-block references. The readout is **merged** into the text (augmenting,
never overwriting, and tagged `[From drawings]` for provenance). The pass is
config-gated (`PLANNING_ENABLE_VISION`) so it can be disabled to save cost, with a text-only fallback or if image support is no longer needed.

### 1.4 Iterative assessment agent (`src/assessment/agent.py`)

A tool-calling loop is given the profile and site constraints plus three tools —
`policy_search`, `policy_lookup`, and `postcode_lookup` — and works a fixed checklist
of material considerations (principle, design, heritage, amenity, highways/parking,
flood/drainage, ecology/trees, contamination). It is prompted to investigate
iteratively: search, read the policy in full before citing it, and if the evidence
is inconclusive refine the query or look a related policy up, rather than guess. For
each theme it emits a decision, the policies cited, a harm level, and a suggested condition where relevant.

### 1.5 Deterministic synthesis (`src/assessment/synthesize.py`)

The decision boundary is deterministic and auditable: **refuse if any theme carries
unresolved harm, otherwise approve with conditions.** Reasons are the per-theme
findings in canonical order. Approvals prepend the two standard conditions (Section 91
three-year commencement and the "approved plans" condition built from the profile's
drawing references), then each theme's suggested condition with its `Reason:` clause.

### 1.6 Guardrails (`src/guardrails.py`)

| Guardrail | What it checks | Strongest action |
|---|---|---|
| scope | assessable housing application with the minimum facts | blocker if no proposal |
| grounding | every cited policy exists in the corpus (verified against the index) | blocker on a hallucinated citation behind a refusal |
| consistency | decision-notice invariants (refusals have no conditions; outcome matches unresolved-harm count) | blocker on violation |
| approval_bias | a statutory objection not turned into refusal/condition | blocker on an unengaged objection |
| abstention | some cited evidence actually exists | blocker if nothing is cited |

Because the deliverable is a draft to be looked at by a human officer, the active intervention is **escalation**:
blockers set `requires_human_review`, surfaced in the officer draft and the eval trace.
The binary prediction is left intact so the JSON is always valid.

### 1.7 Per-component model choices 

Model selection is made **per component** in `ModelConfig`, because the components
differ in difficulty, and every value is env-overridable:

| Component | Default | Rationale |
|---|---|---|
| `profile_model` | `openai/gpt-4o-mini` | extraction from clean text — should be easier, can be cheap |
| `vision_model` | `openai/gpt-4o-mini` (vision-capable) | reads rasterised CAD; should probably be a more complex model but thinking about $25 budget |
| `reasoning_model` | `openai/o3` | the hard, high-stakes judgment step gets a dedicated reasoning model |
| `embed_model` | `BAAI/bge-small-en-v1.5` (local) | runs offline; no per-query cost; |
| judge (eval only) | `openai/o3` | strong, separate reasoning model to grade reasons semantically  |

The cheap `mini` models are kept for the easy extraction/vision steps while the hard
judgment and evaluation steps use `openai/o3`. Because o-series models don't take a
`temperature` argument and ignore `seed`, `sampling_params()` automatically omits those
parameters for them (detected via `is_reasoning_model`) and lets the model manage its own
internal sampling; the provider-pin still applies. Embeddings are intentionally local and
will not be "upgraded" to a hosted model.

## 2. Evaluation 

Evaluation outputs can be found in`evaluation/`.

### 2.1 What we measure, and why

`scripts/evaluate_dev.py` runs the **full end-to-end pipeline** for each labelled case
and writes a timestamped `evaluation/runs/<ts>/` containing `metrics.json`,
`error_analysis.md`, and a per-case `cases/<id>.json` trace (prediction, ground truth,
scores, profile, per-theme assessment, guardrail report).

The **headline metric is end-to-end decision accuracy** — did we return the correct
`approve`/`refuse`. Alongside it I track a **confusion matrix**
(expected → predicted) to expose any directional biases and **reasoning F1** (LLM-judge semantic match of the reasons) as a secondary signal.

### 2.2 Methodology and guarding against overfitting

Only five cases are labelled (`001`–`005`), so a single split would be super prone to
overfitting. To try and work around this, I used Leave-One-Out Cross-Validation. The prompt is written to be outcome-neutral (refusal is first-class, objections must be engaged) rather than
calibrated to the common "approve with conditions" result, and the approval-bias
guardrail is the automated backstop for that lean.

### 2.3 Error analysis

`error_analysis.md` is generated within `evaluation/runs/<run>/`each run with the per-case diff. The failure modes I
specifically inspect:

- **Borderline gate flips** — a `conditionable`/`unresolved` miscall flipping the
  whole decision; the single biggest driver of decision-accuracy loss.
- **Approval bias** — the confusion matrix's `refuse→approve` cell; the guardrail
  should be firing on these.
- **Citation drift** — plausible-but-wrong policy numbers; the grounding guardrail
  quantifies these against the index.
- **Missed themes / recall** — a material consideration never raised; mitigated by the
  fixed checklist.

## 3. How to run

```bash
uv sync                                  # install deps (fastembed, pymupdf, …)
uv run assignment-setup                  # decrypt OpenRouter creds into .env
uv run parser                            # one-time PDF text cache (data/extracted)
uv run planning build-index              # one-time policy embedding index (cached)
uv run python scripts/evaluate_dev.py    # score cases 001–005; writes evaluation/runs/
uv run pytest -q                         # deterministic unit tests (no API)
uv run planning batch                    # generate predictions for holdout 006–010
uv run python scripts/validate_submission.py
```


