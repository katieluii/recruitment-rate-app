# Project Memory State

## Current Context

WSi is the clinical-trial duration and recruitment-rate predictor at
`https://web-production-e6859b.up.railway.app`. `main` and `origin/main` are at
`f970f0e` (`Add direct trial-history recruitment rate model`). Railway deployment
`101d8803-3384-4a59-b73e-9e458d034838` completed successfully from that commit.
The only local untracked path is `.analysis-harness/`.

WSi v5 remains the selected duration model: a refit random-forest point estimate,
forest-shaped split-conformal interval and retained two-stage recruitment/follow-up
split. The new direct recruitment-rate model is a separate Tier B record-history
head for P1/P2/P3. WS21 pins this exact WSi commit and consumes both outputs.

## Completed

- Added a direct target in patients per centre per month: actual enrolment divided
  by initiated centres and the recorded recruiting interval.
- Added P1/P2/P3 history-rate models and artifacts. Their temporal holdouts beat
  therapeutic-area median baselines with skill +0.285/+0.251/+0.462 and achieve
  0.818/0.884/0.829 coverage for nominal 80% intervals.
- Recomputed endpoint/outcome-dependent features after defaults are hydrated, so
  inference inputs remain internally coherent.
- Preserved the validated v5 duration model and its mature-horizon metrics.
- All 106 tests passed before release; commit `f970f0e` was pushed and Railway
  deployed it successfully.

## Known Issues

- The direct rate target is Tier B registry reconstruction, not observed
  centre-level performance. It assumes listed initiated centres were available
  throughout the recorded recruiting interval.
- P1HV has no released direct recruitment-rate head because the current evidence
  does not clear the same data and validation gates.
- Rate uncertainty remains material: temporal median factor error is roughly
  1.54–1.59× across P1/P2/P3.
- `.analysis-harness/` is intentionally local and untracked.

## Exact Next Steps

1. Keep v5 as the duration release until a fully re-evaluated candidate beats its
   mature-horizon results and coverage gates.
2. If improving rate quality, prioritise Tier A facility-status histories that can
   integrate active centre-months rather than weakening the current target gate.
3. Revisit P1HV only when enough defensible record-history targets are available.
4. Preserve the ledger and temporal holdout checks for every future model change.
