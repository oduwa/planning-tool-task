# Evaluation

This directory holds the **evidence** of evaluation-driven development, and it is
deliberately committed (see the negations at the bottom of `.gitignore`) so that
reviewers can see it. Traces and logs left out of a submission cannot be judged.

## How to run

```bash
uv run parser                 # 1. one-time PDF text cache
uv run planning build-index   # 2. one-time policy embedding index
uv run python scripts/evaluate_dev.py   # 3. score all labelled cases (001-005)
```

Each run writes a timestamped directory under `runs/`:

```
runs/<timestamp>/
  metrics.json        end-to-end task metrics + confusion matrix + model config
  error_analysis.md   per-case diff, mismatches and guardrail flags
  cases/<case>.json   full per-case trace: prediction, ground truth, scores,
                      profile, per-theme assessment, guardrail report
```

## What we measure, and why

The **headline metric is end-to-end decision accuracy** (did the tool return the
correct `approve`/`refuse` outcome), not a component score. Strong component
metrics routinely coexist with poor task performance, so the harness scores the
actual deliverable. Alongside it we track:

- a **confusion matrix** (expected → predicted) to expose directional bias — in
  planning, a systematic lean toward `approve` is a known failure mode;
- **reasoning F1** (semantic match of the reasons to the ground-truth reasons,
  via an LLM judge) as a secondary, component-level signal;
- the **guardrail report** per case, so we can see whether the safeguards fire on
  the cases they should.

## Methodology notes

- Only five cases are labelled (`001`–`005`), so a single split is noisy. We use
  a **leave-one-out** discipline during prompt/retrieval iteration: tune against
  four cases, check the held-out fifth, rotate. This is the closest available
  proxy for the unseen holdout (`006`–`010`) and guards against overfitting the
  prompt to the majority historical outcome.
- Outputs on a hosted API are not bit-reproducible (provider routing, batched
  GPU non-determinism). We reduce variance with `temperature=0`, a fixed `seed`,
  and an optional OpenRouter provider pin (`PLANNING_PROVIDER_ORDER`), and we
  record the model config in every `metrics.json` for provenance.

## Committed run artifacts

Populated `runs/<timestamp>/` directories in this folder are real evaluation
outputs from development and are part of the submission evidence.
