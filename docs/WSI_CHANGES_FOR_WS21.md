# WSi changed under you — what WS21 needs to know

Written 2026-08-05 by the WSi session, for the WS21 session running concurrently.
WS21 imports WSi in-process (`analyst/wsi.py` puts this checkout on `sys.path`), so
everything below is ALREADY in effect for you — there is nothing to install.

**WS21 pins WSi at `b85f65b7aacf`. WSi's HEAD is now `daf2e90a`.** `check_pin()` warns
and never fails, so a run against these changes looks normal apart from one log line.

## What invalidates work you may already have done

**1. The corpus was roughly half its true size until 2026-08-04.**
`cleaner.parse_dates` parsed a column holding both `2015-10` and `2022-10-21` without
an explicit format. Pandas inferred one format from the first non-null value and the
`dropna` two lines later deleted everything that did not match — silently, and the
half that died depended on the order the API happened to return.

| phase | rows before | rows now |
|---|---|---|
| P1 | 8,768 | 15,429 |
| P1HV | 4,102 | 7,416 |
| P2 | 7,726 | 16,126 |
| P3 | 7,847 | 14,959 |

**Any clustering, stratification or frequency count you computed before today used
roughly half the trials, biased by era.** P3 was the worst: it lost its NEWER records
and had no post-2017 starts at all. If WS21 has endpoint clusters, criteria clusters
or enrolment strata on disk, they need recomputing.

**2. The endpoint classifier was wrong for 893 trials, and you use it directly.**
`EVENT_COMPOSITE`'s regex ended `incidence of .{0,40}(?:events?|episodes?)`, which
matches "Incidence of treatment-emergent adverse events" — the most common safety
phrasing in the registry. EVENT_COMPOSITE is rule 2 of 10 and SAFETY is rule 7, so
first-match-wins never reached SAFETY. 893 of 1,006 EVENT_COMPOSITE trials were
safety trials, 100% of them on P1 and P1HV. Fixed with a negative lookahead; genuine
composites (MACE, all-cause mortality, hospitalisation-for) still classify.

P1 `EVENT_COMPOSITE` fell 480 → 2, and SAFETY rose correspondingly. **Any endpoint
cluster built before today has that mislabel baked in.**

**3. The abstention rates in your docs are stale.** `WS21_KICKOFF.md` and the tracker
card quote 8.6 / 21.0 / 23.9% for P1 / P2 / P3. Measured on the recovered corpus:
**8.9 / 21.0 / 33.5%**. P3 is 10 points worse than recorded, which matters if you
sized a clustering approach against the old figure.

**4. Scoring changed.** The absolute R2 >= 0.70 gate is retired (Katie's call — it was
unreachable from a feature set with no per-site enrolment and no country speed). R2
and RMSE are optimisation targets against each phase's own best, held in
`experiments/leaderboard.py`. `experiments/metrics.py:GATES` no longer has an
`r2_min` key — WS21 does not read it today, but your vendored copy of that module has
drifted further from WSi's. Default split is now `horizon` (train <2018 / test
2018-2020), because the old 2021+ fold was truncated by observation horizon and
rewarded any change that merely predicted shorter.

**5. All four models were retrained** on the recovered corpus and are live. Served
durations moved: P3 Cardiovascular -9.9 months, P2 Oncology -6.6, most others -2 to
0, P3 Oncology +2.8. If WS21 has fixtures asserting WSi's duration output, they will
fail, and the new numbers are the correct ones.

## The known-good direction of travel

On the honest fold WSi **under-predicts in every test year on every phase** — P1 by
2.8-3.5 months, P2 by 2.5-4.8, P3 by 2.3-5.9. If WS21 surfaces WSi's duration to a
user, it is currently optimistic by roughly that much. Worth a caveat in the read
surface until it is corrected.

## Ownership, agreed 2026-08-05

The dividing test Katie set: does it feed the recruitment-rate / duration prediction?

**WSi keeps the endpoint CLASSIFIER**, because it feeds the prediction three ways —
`endpoint_archetype` is a model feature, the ten `endpoint_has_*` flags are features,
and `cleaner.py` imputes `followup_months` from the archetype median, which sets the
enrolment/follow-up split the two-stage model trains on. It partly defines the label,
not just the input.

**WS21 owns the endpoint PROFILES layer** — most common endpoint combinations per
indication x phase, with frequencies and observed medians. Nothing in WSi's model or
preprocessing reads it; the registry only carries it alongside the artifacts.

WSi has a working implementation at `backend/analytics/endpoint_profiles.py`
(`build_profiles(df)` → `{therapeutic_area: [{archetypes, label, n, share,
median_months}]}`, top 3, cells under 30 trials falling back phase-wide and saying
so). It is precomputed into `models/artifacts/<phase>/endpoint_profiles.json`.
**Import it through your bridge rather than reimplementing** — or take it over
wholesale, WSi has no use for it beyond serving.

One finding from building it, worth carrying: **a strict top-3 hides the most
decision-relevant oncology case.** On the full corpus P1 oncology ranks SAFETY
(n=2,101, 33.0 mo), PK_PD (455, 11.0), UNCLASSIFIED (344, 26.0), and only then
RESPONSE+SAFETY (330, **44.7 mo**) — which runs ~12 months longer than SAFETY alone.
Ranking purely by frequency buries it behind the unclassified bucket.

Also: trials list 4.15 primary outcome measures that collapse to 1.17 distinct
archetypes, and 21.8% carry more than one. The combination is not the sum of its
parts, so the unit worth clustering on is the combination, not the single label.

**The interactive UI work is WS21's**, per `WS21_KICKOFF.md` and the tracker card —
dropping the free-text enrolment box, replacing the archetype dropdown with real
endpoints. WSi has a spec for it in `SPECS.md` written before that was settled; it is
marked WS21-owned and WSi is not building it.

**One thing WSi should still fix in its own API**, because it is a WSi model bug:
`inference.py` sets exactly ONE `endpoint_has_*` flag while the model trains on a
multi-hot set, so a RESPONSE+SAFETY trial cannot be expressed through `/predict` and
comes back as the shorter of its halves. Not done yet.

## Suggested next step for WS21

1. Recompute any clusters, strata or frequency counts — the inputs changed twice.
2. Re-run your fixtures against WSi's new duration output; the movements above are
   expected, not regressions.
3. Bump `PINNED_COMMIT` to `daf2e90a0da8` once you have re-verified agreement, so the
   pin means something again.
