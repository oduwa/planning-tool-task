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

The packs are deliberately messy: every document appears twice (hyphenated and
underscored) and some ship superseded/refused/withdrawn variants. Files are grouped
by a normalised stem with any superseded marker stripped, and one survivor is chosen
per group — always preferring a live version over a superseded one. Every dropped
file is recorded for the audit trail. Survivors are classified (application form,
design & access, heritage/flood statements, elevations, site/floor plans,
consultation replies), which drives both the assembled text and the later weighting
of consultee objections.

PDF text is read exclusively from the JSON that the provided `uv run parser` writes
under `data/extracted` — persistent, reused across every run, with no live re-parsing.
A missing cache raises a clear, actionable error rather than silently re-parsing.

### 1.2 Full-corpus policy retrieval (`src/policy/`)

The policy corpus (NPPF, primary legislation, the Doncaster Local Plan, six SPDs,
conservation-area appraisals, the town-centre masterplan, the HMO Article 4
direction) is far too large for a prompt, and the brief describes officers making
"multiple lookups", so this is a retrieval problem. **The index walks the entire
national and local policy trees** (`rglob("*.pdf")`), so coverage is the whole corpus,
not a hand-picked subset. Each PDF is chunked into overlapping windows tagged with any
detectable policy reference (e.g. "Policy 44", "NPPF paragraph 135"), embedded with a
**local** model (`fastembed`, `BAAI/bge-small-en-v1.5`), and cached to disk as a numpy
matrix plus a JSONL sidecar — **persistent indexing** built once and reused.

Retrieval is not a single-pass top-k lookup:

- **`search`** re-ranks candidates with **Maximal Marginal Relevance** so repeated
  queries on a theme return *diverse* passages (different policies/documents) rather
  than near-duplicates — which is what makes iterative deepening add real evidence.
- **`lookup_reference`** pulls every passage that mentions a specific named policy or
  paragraph, so a citation can be read in full before it is relied on, and so a claimed
  policy can be checked for existence.

### 1.3 Multimodal plan reading (`src/assessment/vision.py`)

The README requires reasoning across "diagrams & measurements". The drawings are
vector CAD sheets with **no embedded raster images**, so the parser's image extraction
returns nothing — a text-only pipeline is effectively blind to geometry. We therefore
**rasterise each drawing page with PyMuPDF** (~144 dpi) and send the PNGs to a vision
model, which recovers ridge/eaves heights, storeys, separation distances, parking and
title-block references. The readout is **merged** into the text profile (augmenting,
never overwriting, and tagged `[From drawings]` for provenance). The pass is
config-gated (`PLANNING_ENABLE_VISION`) so it can be disabled to save cost, with a
graceful text-only fallback.

### 1.4 Iterative assessment agent (`src/assessment/agent.py`)

A tool-calling loop is given the profile and site constraints plus three tools —
`policy_search`, `policy_lookup`, and `postcode_lookup` — and works a fixed checklist
of material considerations (principle, design, heritage, amenity, highways/parking,
flood/drainage, ecology/trees, contamination). It is prompted to **investigate
iteratively**: search, read the policy in full before citing it, and if the evidence
is inconclusive refine the query or look a related policy up, rather than guess. For
each theme it emits a decision-notice-style finding, the policies cited, a harm level
(`none` / `conditionable` / `unresolved`), and a suggested condition where relevant.

### 1.5 Deterministic synthesis (`src/assessment/synthesize.py`)

The decision boundary is deterministic and auditable: **refuse if any theme carries
unresolved harm, otherwise approve with conditions.** Reasons are the per-theme
findings in canonical order. Approvals prepend the two standard conditions (Section 91
three-year commencement and the "approved plans" condition built from the profile's
drawing references), then each theme's suggested condition with its `Reason:` clause.

### 1.6 Guardrails (`src/guardrails.py`)

Five substantive guardrails read real pipeline state (profile, assessment, emitted
decision, and the **live** policy index) — not regex over the output:

| Guardrail | What it checks | Strongest action |
|---|---|---|
| scope | assessable housing application with the minimum facts | blocker if no proposal |
| grounding | every cited policy exists in the corpus (verified against the index) | blocker on a hallucinated citation behind a refusal |
| consistency | decision-notice invariants (refusals have no conditions; outcome matches unresolved-harm count) | blocker on violation |
| approval_bias | a statutory objection not turned into refusal/condition | blocker on an unengaged objection |
| abstention | some cited evidence actually exists | blocker if nothing is cited |

Because the deliverable is a *draft*, the active intervention is **escalation**:
blockers set `requires_human_review`, surfaced in the officer draft and the eval trace.
The binary prediction is left intact so the JSON is always valid.

### 1.7 Per-component model choice — interrogated, not decided once

Model selection is made **per component** in `ModelConfig`, because the components
differ in difficulty, and every value is env-overridable:

| Component | Default | Rationale |
|---|---|---|
| `profile_model` | `openai/gpt-4o-mini` | structured extraction from clean text — easy, cheap |
| `vision_model` | `openai/gpt-4o-mini` (vision-capable) | reads rasterised CAD; upgrade to a larger vision model for hard sheets |
| `reasoning_model` | `openai/o3` | the hard, high-stakes judgment step gets a dedicated reasoning model |
| `embed_model` | `BAAI/bge-small-en-v1.5` (local) | runs offline; no per-query cost; OpenRouter has no reliable embeddings endpoint |
| judge (eval only) | `openai/o3` | strong, separate reasoning model to grade reasons semantically without grading itself |

The cheap `mini` models are kept for the easy extraction/vision steps while the hard
judgment and evaluation steps use `openai/o3`. Because o-series reasoning models reject a
custom `temperature` and ignore `seed`, `sampling_params()` automatically omits those
parameters for them (detected via `is_reasoning_model`) and lets the model manage its own
internal sampling; the provider-pin still applies. Embeddings are intentionally local and
will not be "upgraded" to a hosted model.

### 1.8 Determinism

Hosted APIs are not bit-reproducible (provider routing, batched-GPU
non-determinism). We reduce variance with `temperature=0`, a fixed `seed`, and an
optional OpenRouter provider pin (`PLANNING_PROVIDER_ORDER`), and record the exact
model config in every `metrics.json`. Residual run-to-run variation lands on genuinely
borderline cases, where one theme sits on the `conditionable`/`unresolved` line and
tips the hard decision gate; this is a property of the task, and the guardrails flag
such cases for review.

## 2. Evaluation (weighted 25%)

Evaluation is treated as a first-class, evidence-producing part of the build, and the
evidence is **committed** under `evaluation/` — a `.gitignore` mistake that hides eval
traces would make them unjudgeable, so the directory is force-included.

### 2.1 What we measure, and why

`scripts/evaluate_dev.py` runs the **full end-to-end pipeline** for each labelled case
and writes a timestamped `evaluation/runs/<ts>/` containing `metrics.json`,
`error_analysis.md`, and a per-case `cases/<id>.json` trace (prediction, ground truth,
scores, profile, per-theme assessment, guardrail report).

The **headline metric is end-to-end decision accuracy** — did we return the correct
`approve`/`refuse` — not a component score, because strong component metrics routinely
mask poor task performance. Alongside it we track a **confusion matrix**
(expected → predicted) to expose directional bias (the known approval-lean failure
mode), and **reasoning F1** (LLM-judge semantic match of the reasons) as a secondary
signal.

### 2.2 Methodology and guarding against overfitting

Only five cases are labelled (`001`–`005`), so a single split is noisy and easy to
overfit — precisely the flagged failure mode of tuning prompts to the majority
historical outcome. During iteration we use a **leave-one-out discipline**: tune
against four, check the held-out fifth, rotate. The prompt is written to be
outcome-neutral (refusal is first-class, objections must be engaged) rather than
calibrated to the common "approve with conditions" result, and the approval-bias
guardrail is the automated backstop for that lean.

### 2.3 Error analysis (to populate from live runs)

`error_analysis.md` is generated each run with the per-case diff. The failure modes we
specifically inspect:

- **Borderline gate flips** — a `conditionable`/`unresolved` mis-call flipping the
  whole decision; the single biggest driver of decision-accuracy loss.
- **Approval bias** — the confusion matrix's `refuse→approve` cell; the guardrail
  should be firing on these.
- **Citation drift** — plausible-but-wrong policy numbers; the grounding guardrail
  quantifies these against the index.
- **Missed themes / recall** — a material consideration never raised; mitigated by the
  fixed checklist.

> The `evaluation/runs/` artifacts committed in this PR are the concrete evidence of
> the above; the headline numbers are reported there rather than hard-coded here so the
> report and the artifacts cannot drift apart.

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

## 4. Strengths and limitations (honest assessment)

**Strengths.** Full-corpus retrieval with diversity and citation verification;
genuinely multimodal (rasterised CAD + vision); a deterministic, auditable decision
boundary; substantive guardrails wired to real state; per-component model choice;
committed evaluation evidence.

**Limitations.**
- **Five labelled cases** cap statistical confidence; leave-one-out is a proxy, not a
  guarantee of holdout performance.
- **Geospatial lookup is hardcoded** to the supplied postcodes (per the boilerplate),
  so an unknown site degrades to "cannot confirm".
- **Hosted-API variance** can still flip a truly borderline case between runs.
- **Vision cost/latency**: the pass adds image tokens; it is capped by a page budget
  and can be disabled.
- **Reasoning model**: the mini default trades accuracy for the restricted key's
  budget; production should upgrade `reasoning_model`.

## 5. What I would do with more time

- Upgrade and A/B the `reasoning_model`, reporting the accuracy/£ trade-off from
  `metrics.json`.
- Add a self-consistency vote on each theme's harm level to stabilise the borderline
  gate.
- Expand the labelled set with officer-reviewed cases and move from leave-one-out to a
  proper held-out split.
- A retrieval-quality harness (does the cited policy appear in the retrieved chunks?)
  to measure grounding independently of decision accuracy.
- A genuine geospatial backend behind `postcode_lookup`.

## 6. Use of coding assistants

Built with the assistance of v0. The architecture, module boundaries, decision logic
(derived from the five worked cases), guardrail design, and prompts were directed and
reviewed by me; the assistant produced the implementation, verified with ruff, mypy,
and the committed `pytest` suite over the deterministic ingestion, synthesis, guardrail,
retrieval-matching, and plan-merge logic.
