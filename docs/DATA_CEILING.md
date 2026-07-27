# Why R² stalls around 0.34–0.51, and what would move it

Written after option (b): testing whether AACT or any other public source carries
information ClinicalTrials.gov's API does not.

## What was tested

**AACT.** It is a mirror of ClinicalTrials.gov, not a superset. Checked against the
published data dictionary, its `ctgov` schema has:

- **no** enrolment counts over time, by date, or as accrual history
- **no** site or facility activation dates
- **no** per-site or per-country enrolment breakdown

Its one genuinely derived table, `calculated_values`, holds fields this project
already computes for itself: `number_of_facilities` is our `site_count`,
`actual_duration` is our `duration_days`. The `milestones` table summarises
participant flow, and it only exists after a trial has posted results, so it is an
outcome rather than a design-time predictor.

**Record history.** The `/api/int/studies/{nct}/history` endpoint does carry
something the main API hides: `originalData`, the record as first registered, and
with it the original enrolment TARGET. That is a real find, and it exposes a
train/serve mismatch.

## The mismatch it exposed

88% of completed trials report `enrollmentInfo.type = ACTUAL`. So the model
largely learns on what a trial ACHIEVED, while a user at design time can only
supply what they PLAN. Sampled across 300 Phase 3 trials (30 resolved before the
endpoint rate-limited):

| | value |
|---|---|
| Original registered type | 93% ESTIMATED |
| Median target vs achieved | 314 vs 328 |
| Achieved / target ratio | median 1.04, IQR 1.00–1.12 |
| Differ by more than 10% | 40% of trials |
| log-log correlation | **0.953** |

Same class of defect as the `site_count` bug, which cost 18 months of duration
error. This one is far smaller: at a correlation of 0.95 the two are nearly the
same feature. Recorded, surfaced in provenance, not worth a corpus-wide refetch
the endpoint would rate-limit anyway.

## The conclusion

**R² 0.70 is not reachable from public registry data.** Not because the model is
weak, but because the information is absent.

One caveat this document originally got wrong, kept here rather than quietly
edited out: it claimed both modelling levers were spent. They were not. Changing
the point head from the alpha=0.5 quantile to a squared-error objective then
gained 0.038 on Phase 1 and 0.027 on Phase 3 for no new data at all. The lesson
is narrow but worth keeping — "the data is the ceiling" was true about the
distance to 0.70 and false about the distance to the next 0.03, and only the
second claim was testable in an afternoon. Test the loss function before
declaring a ceiling.

What actually determines whether a trial finishes on time:

- how many planned sites ever activate, and when
- how fast each one enrols once open
- which competing trials are recruiting the same patients
- protocol amendments mid-flight
- interim analyses, and the sponsor's willingness to keep funding

None of it appears in a registry record. The registry describes what a trial
INTENDED and, afterwards, what it achieved in total. It never describes the
process in between, and the process is the thing being predicted.

This is consistent with the literature: work on this exact problem reports
ranking accuracy (C-index around 0.78) rather than R², because ranking is what
the available features support.

## Where the ceiling actually sits

Best after 5× more training data, a hyperparameter search, and an L2 point head,
all on one temporal fold:

| phase | R² | RMSE (days) |
|---|---|---|
| P1 | 0.555 | 226 |
| P1HV | 0.370 | 140 |
| P2 | 0.345 | 280 |
| P3 | 0.372 | 278 |

Three levers now spent. Lifting the API cap moved P3 more than tuning did; tuning
converged on a smaller, slower learner with only 3 of 18 trials beating the
defaults, which is the signature of a flat space and a modest signal; and the L2
head took the remainder. None of the three closes even half the gap to 0.70.

P1HV is the one phase where v1's RandomForest still scores higher on R² (0.420
against 0.370), and it does so for the same reason the L2 head helped elsewhere:
averaging over trees estimates a mean. It loses on MAE. The two metrics disagree
on that cohort and the disagreement is real, not a bug.

## What would actually move it

1. **A source with per-site or per-period enrolment.** Citeline, TrialTrove, or a
   sponsor's own CTMS. This is a purchase decision, not an engineering one, and it
   is the only route to the numbers a feasibility team would call reliable.
2. **Predict the right thing.** If the goal is choosing between options rather
   than quoting a date, optimise ranking (C-index) instead of R². The features
   support ranking better than they support point accuracy.
3. **Say less, more honestly.** A calibrated interval that covers 82–86% of
   outcomes is already useful for planning; a point estimate with ±9 months of
   error is not, and dressing it up would be worse than admitting it.

The 0.70 gate stays in `experiments/metrics.py` and everything reports as failing
against it. Moving the bar to make the numbers green would defeat the point of
having recorded a baseline in the first place.
