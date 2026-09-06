# Project Memory State

## Current Context

WSi is the clinical-trial duration and recruitment-rate predictor at
`https://web-production-e6859b.up.railway.app`. `main` and `origin/main` are both at
`99c4668` (`Client copy gate: land check_client_copy.py on main and clear 83 violations`);
divergence check reads `0	0` (ahead/behind). The only local untracked path is
`.analysis-harness/`, which is intentionally local.

WSi v5 remains the selected duration model: a refit random-forest point estimate,
forest-shaped split-conformal interval and retained two-stage recruitment/follow-up
split. The direct recruitment-rate model is a separate Tier B record-history head for
P1/P2/P3. WS21 pins a WSi commit and consumes both outputs.

## Completed

- House-style gate `scripts/check_client_copy.py` now lives on `main`. It scans the
  frontend, backend routes/models/analytics and top-level markdown for em dash, middle
  dot and AI-slop phrasing. It exits 0 on the current tree.
- `tests/test_client_copy.py` carries two independent layers: an AST walk over
  `publish_metrics` string literals, rendered markdown and authored docs, plus scanner
  tests including a planted-violation mutation test, so the gate can demonstrably fail.
- 83 violations cleared across 10 files. Four files untouched since the branch point took
  the reviewed wording from `claude/no-ai-punctuation`; six were rewritten against
  current v5 content.
- Numeric content verified unchanged: each of the 10 files yields an identical number
  sequence before and after. Full suite: 111 passed.
- The abandoned analysis pass `wsi-v5-selection-for-ws21` was closed out. State archived
  to `.analysis-harness/closed-20260906/` with a `CLOSURE.md`; nothing deleted.
- Earlier release state retained: direct P1/P2/P3 history-rate models with skill
  +0.285/+0.251/+0.462 and 0.818/0.884/0.829 coverage for nominal 80% intervals.

## Known Issues

- `claude/no-ai-punctuation` is superseded but is NOT an ancestor of `main`; the gate was
  re-implemented against current content rather than merged. The branch is deliberately
  kept, not deleted.
- The direct rate target is Tier B registry reconstruction, not observed centre-level
  performance. It assumes listed initiated centres were available throughout the recorded
  recruiting interval.
- P1HV has no released direct recruitment-rate head; current evidence does not clear the
  same data and validation gates.
- Rate uncertainty remains material: temporal median factor error is roughly 1.54 to 1.59x
  across P1/P2/P3.
- Published figures are still measured on a fold that does not match the shipped model
  configuration; re-publishing is an open decision, not a completed task.

## Exact Next Steps

1. Keep v5 as the duration release until a fully re-evaluated candidate beats its
   mature-horizon results and coverage gates.
2. Decide whether `claude/no-ai-punctuation` can now be deleted, given its content is
   superseded but not contained in `main`.
3. If improving rate quality, prioritise Tier A facility-status histories that can
   integrate active centre-months rather than weakening the current target gate.
4. Revisit P1HV only when enough defensible record-history targets are available.
5. Preserve the ledger and temporal holdout checks for every future model change.
