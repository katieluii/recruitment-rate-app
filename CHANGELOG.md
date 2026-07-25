# Changelog

## v2.0 — 2026-07-25 · Model rebuild, site-level rates, endpoint stratification

Branch `claude/v2-recruitment-model`. Revert point: tag `v1.0-baseline`.

### The problem

v1 returned near-identical durations for most therapeutic areas. Measured across
all 22 areas, the shipped models reproduced roughly **6% of the between-area
signal present in their own training data**:

| Phase | True spread across areas | v1 predicted spread | Distinct predictions |
|---|---|---|---|
| P1 | 7.0 mo | 1.3 mo | 5 of 22 |
| P2 | 21.5 mo | 4.4 mo | 6 of 22 |
| P3 | 28.4 mo (onc 38.0 vs derm 9.5) | 1.6 mo | 9 of 22 |

Deeper: with no baseline ever recorded, nobody knew the model beat a lookup
table. On a temporal holdout it did not — it was **2.9× worse than a per-therapeutic-area
median** on P2 and P3.

### Root causes, each confirmed by experiment

1. **`site_count` counted countries, not sites.** Training values sat in 1–20;
   inference passed real site counts (P3 default 40, API allowed 5000). Every
   prediction was evaluated outside the trained range, where a random forest is
   flat by construction. Real site count was being fetched and discarded.
2. **`primary_completion_year` leaked the label's own endpoint.** Removing that
   single feature took P2 MAE from **25.41 → 8.88 months** (bias +25.3 → +4.1).
3. **22 sparse therapeutic-area binaries could not compete** with six scaled
   continuous features inside the forest.
4. **The target fused two processes** — recruitment and follow-up. Oncology
   Phase 3's length is follow-up, not slow enrolment.
5. **No baseline, no cross-validation, one metric**, and an interval whose
   half-width was pinned at `rmse * 0.5` for every input (8% coverage on P2).

### What changed

- **Experiment harness** (`experiments/`) — temporal split, three median
  baselines, append-only ledger, per-area error tables, and two metrics aimed at
  the failure: predicted-vs-true area spread, and area rank correlation.
- **Data layer** — real site count, retained site list, endpoint measure and
  time-frame text, eligibility restrictiveness, arm count, sponsor. Calendar year
  removed entirely. Winsorising replaces hard caps that deleted the long tail.
- **Endpoint archetype classifier** — 11-value closed vocabulary, deterministic
  rules, 8.6 / 21.0 / 23.9% abstention. On P3 the archetype spans **29 months** of
  median duration on its own (immunogenicity 10.6 → survival 39.6).
- **Therapeutic-area target encoding** — leak-free, fitted in-pipeline with
  out-of-fold encoding inside the training fold.
- **Two LightGBM heads** replacing the forest: duration on `log1p`, recruitment
  rate on `log`. Three quantile models each, widened by conformal calibration on
  the most recent training slice.
- **Site-level layer** — country × area rate priors with shrinkage, facility
  track record, site-mix simulator, `/api/site-rates/*`.
- **Guards** — extrapolation warnings when an input leaves the trained range, and
  a regression suite that fails if areas collapse, intervals go constant-width,
  oncology is predicted faster than dermatology, or a year feature returns.

### Results — 2,039 real held-out trials

Both recipes refit on pre-2021 trials, compared on real post-2021 studies neither
model was shown.

| | v1 | v2 |
|---|---|---|
| Mean absolute error | 7.18 mo | **5.91 mo** |
| Trials called closer | — | **61.5%** |
| Actuals inside 80% interval | — | **83.2%** |

Predicted median by therapeutic area against reality:

| Area | Actual | v1 | v2 |
|---|---|---|---|
| Oncology | 21.6 | 31.5 | **26.2** |
| Haematology | 17.9 | 31.0 | **26.8** |
| Dermatology | 13.4 | 19.0 | **14.1** |
| Metabolic | 11.2 | 16.6 | **11.7** |
| Infectious Diseases | 10.6 | 17.5 | **11.8** |

v1 over-predicted **every** area by 5–10 months. Skill against the per-area median
baseline went from −1.87 to +0.21 on P2 and −1.72 to +0.27 on P3; the recruitment
rate head reaches +0.54 on P3. Interval coverage went 0.08 → 0.82 on P2.

App-level area differentiation: P3 spread **1.6 → 18.9 months**, distinct
predictions **9 → 19 of 22**, ordered correctly.

### Known limitations

- No observed per-site enrolment exists in ClinicalTrials.gov or AACT. Site-level
  output is modelled; facility figures are association, not attribution.
- The recruitment rate's denominator is the full start-to-primary-completion span,
  so long-follow-up trials have their rate understated.
- The corpus is completed trials only, so recent history skews fast.
- v2 still over-predicts oncology by ~4.6 and haematology by ~9 months at the
  median. Largest remaining gap.
- Phase 1 healthy-volunteer is a wash (2.82 vs 2.76 mo) — short, uniform trials
  with little signal to exploit.

---

## v1.0 — 2026-05-28 · Initial release

FastAPI + per-phase RandomForest over ClinicalTrials.gov, Plotly error bars,
Railway deploy. Tagged `v1.0-baseline`.
