# Clinical Trial Duration Predictor

> A full-stack ML web app that predicts how long a clinical trial will take,
> powered by live data from [ClinicalTrials.gov](https://clinicaltrials.gov)
> and phase-specific LightGBM models with calibrated prediction intervals.

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-brightgreen)
![ClinicalTrials.gov](https://img.shields.io/badge/Data-ClinicalTrials.gov-blue)

---

## What it does

Select a **trial phase** (Phase 1 HV / Phase 1 / Phase 2 / Phase 3) and a
**therapeutic area** (Oncology, CNS, Cardiovascular, …) — the app returns:

- **Predicted trial duration** (months to primary completion)
- **80% prediction interval** (error bar)
- **Interactive analytics charts**:
  - Duration distribution by therapeutic area for the selected phase
  - Phase-comparison bar chart

---

## Architecture

```
ClinicalTrials.gov API  ──►  data layer
         │                        │
         │              dataset builder + cleaner
         │                        │
         │              feature engineering
         │            (TA mapping, region, SAD/MAD)
         │                        │
         │              LightGBM pipeline (per phase)
         │                        │
         ▼                        ▼
      FastAPI backend  ◄──────────┘
           │
     REST API  /api/*
           │
     Static HTML/JS frontend (Plotly)
```

### Module map

```
recruitment_rate_app/
├── backend/
│   ├── main.py                 FastAPI app + static file serving
│   ├── config.py               Pydantic settings (env vars)
│   ├── constants.py            Phase definitions, TA list, defaults
│   ├── data/
│   │   ├── ct_api_client.py    ClinicalTrials.gov API v2 client
│   │   ├── postgres_client.py  AACT PostgreSQL fallback
│   │   └── data_layer.py       Unified data access
│   ├── preprocessing/
│   │   ├── cleaner.py          Date parsing, outlier removal
│   │   ├── features.py         TA & region mapping, SAD/MAD tagging
│   │   └── pipeline.py         sklearn ColumnTransformer pipeline
│   ├── models/
│   │   ├── trainer.py          Train + save models
│   │   ├── registry.py         Load + cache trained pipelines
│   │   └── inference.py        Predict + uncertainty estimation
│   └── routes/
│       ├── meta.py             GET /api/phases, /api/therapeutic-areas
│       ├── predict.py          POST /api/predict
│       └── analytics.py        GET /api/analytics
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js              Fetch wrapper
│       ├── charts.js           Plotly renderers
│       └── app.js              App logic + event handlers
├── models/artifacts/           Trained model files (git-ignored)
│   └── {P1HV,P1,P2,P3}/
│       ├── model.pkl
│       ├── metadata.json
│       └── analytics.json
└── scripts/
    └── train_models.py         One-shot training CLI
```

---

## Quick start

### 1. Install dependencies

```bash
cd recruitment_rate_app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env if you want to use PostgreSQL instead of the CT.gov API
```

### 3. Train models

```bash
# Fetches live data from ClinicalTrials.gov and trains 4 phase-specific models.
# Takes ~10–20 minutes total (API rate-limited). Re-run anytime to update.
python -m scripts.train_models

# Or train a single phase:
python -m scripts.train_models --phase P2
```

### 4. Run the server

```bash
uvicorn backend.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/phases` | List available trial phases and training status |
| `GET`  | `/api/therapeutic-areas` | List all supported therapeutic areas |
| `POST` | `/api/predict` | Predict trial duration + recruitment rate |
| `GET`  | `/api/endpoint-archetypes` | Endpoint-type vocabulary for the optional input |
| `POST` | `/api/site-rates/simulate` | Enrolment projection for a country/site mix |
| `GET`  | `/api/site-rates/countries?phase=P3` | Countries ranked by modelled rate |
| `GET`  | `/api/site-rates/facilities?phase=P3` | Facility track record |
| `GET`  | `/api/analytics?phase=P2` | Analytics data for charts |
| `GET`  | `/api/health` | Server + model status |

### POST /api/predict

```json
{
  "phase": "P2",
  "therapeutic_area": "Oncology/Solid Tumours",
  "enrollment": 120,
  "num_sites": 15,
  "drug_type": "DRUG",
  "region": "US",
  "endpoint_archetype": "SURVIVAL"
}
```

`enrollment`, `num_sites` and `endpoint_archetype` are optional. Left out, each
falls back to the median for that **therapeutic area** — asking for "a Phase 3
oncology trial" assumes an oncology-shaped trial (median 502 patients across 89
sites), not a phase-average one (334 across 32).

Response:

```json
{
  "predicted_months": 28.4,
  "lower_months": 20.6,
  "upper_months": 60.6,
  "predicted_days": 864.5,
  "confidence_pct": 80,
  "recruitment_rate": 0.215,
  "recruitment_rate_lower": 0.078,
  "recruitment_rate_upper": 0.377,
  "rate_implied_total_months": 26.0,
  "rate_note": "Patients per site per month, modelled — no per-site enrolment is published …",
  "model_used": "LightGBM conformalised quantile",
  "n_train": 2024,
  "extrapolation_warnings": []
}
```

`extrapolation_warnings` is populated when an input falls outside the range the
model was trained on. A prediction with warnings is an extrapolation, not an
estimate.

---

## Data sources

| Source | Usage |
|--------|-------|
| [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) | Primary — no account needed |
| [AACT PostgreSQL](https://aact.ctti-clinicaltrials.org) | Optional fallback — free account required |

Only **completed, industry-funded, interventional drug/biological trials**
are included. Trials with missing start/completion dates are excluded.

---

## Model details

Two heads per phase, because duration and recruitment speed are different
processes. Total duration fuses recruitment with follow-up, and follow-up is what
makes an oncology Phase 3 run a median 34 months while dermatology runs 15.

| Head | Target | Transform |
|------|--------|-----------|
| **A — duration** | days, start → primary completion | `log1p` |
| **B — recruitment rate** | patients per site per month | `log` |

Each head is three LightGBM quantile models (α = 0.1 / 0.5 / 0.9). The median is
the point estimate; the outer pair form an **80% prediction interval**, widened
by **conformal calibration** on the most recent slice of training data. Raw
quantiles covered ~0.58 of a nominal 0.80; calibrated they cover 0.78–0.85.

### Features

Trial design (allocation, masking, purpose, arms), scale (enrolment, real site
count, country count), **endpoint archetype** parsed from the primary outcome
text, **eligibility restrictiveness** (age span, inclusion/exclusion counts),
therapeutic area as both one-hots and a leak-free target encoding, and region.

**Calendar year is deliberately excluded.** See the note in
`backend/preprocessing/pipeline.py`.

### Performance

<!-- published_metrics:start — generated by experiments/publish_metrics.py from the ledger; do not hand-edit -->
| Phase | duration MAE | skill | R² | coverage (nominal) | gate |
|-------|--------------|-------|----|--------------------|------|
| P1HV | 3.26 mo | +0.27 | 0.333 | 0.79 (0.85) | pass |
| P1 | 7.71 mo | +0.35 | 0.645 | 0.78 (0.80) | pass |
| P2 | 9.52 mo | +0.22 | 0.423 | 0.79 (0.80) | pass |
| P3 | 10.09 mo | +0.22 | 0.392 | 0.77 (0.80) | pass |

Fold: horizon fold — train on trials starting before 2018, test on 2018–2020 starts, which have had 5.4–8.6 years to finish against a corpus whose p95 duration is 5.9. Rows P1HV=350, P1=336, P2=339, P3=340 of `experiments/ledger.jsonl`. Regenerate: `python -m experiments.publish_metrics`. Every row measures the shipped configuration — IPCW censoring weights applied, as in `trainer.train_phase`.
<!-- published_metrics:end -->

Until 2026-08-29 this table quoted the 2021+ temporal fold (train before 2021, test after).
That fold cannot contain a long trial — the corpus is completed trials only, so a trial
starting in 2022 that runs six years is not in it — and it rewarded anything that predicted
shorter. The horizon fold above is the honest one; its numbers are worse and they are the
claim. `--split temporal` still reproduces the old rows, which are not comparable.

Reproduce: `python -m experiments.run --config two_stage_l2_ipcw --phases P1,P2,P3 --split horizon`
and `--config two_stage_l2_cov85_ipcw --phases P1HV`, then `python -m experiments.publish_metrics`
to refresh both tables from the ledger. The `_ipcw` configs build their censoring frame with
`trainer.build_censoring_frame`, the same function training uses — until 2026-08-30 the eval
configs passed no frame, so the table measured an unweighted model that was not the one serving.
See `experiments/README.md` for the protocol and the v1 comparison.

Recruitment rate (patients per site per month), same fold. What the API serves as the
rate is NOT the rate head's prediction: since Task 13 it is derived from the duration
head's enrolment window — enrollment / (sites × window) — with the band inverted from the
duration band, so the served rate is scored as exactly that (`DerivedRate` in the harness
mirrors `inference.py` line for line). The standalone rate head reaches a response only as
`recruitment_rate_crosscheck`, a point without its band, and is published below as a
cross-check rather than as the rate figure:

<!-- published_metrics_rate:start — generated by experiments/publish_metrics.py from the ledger; do not hand-edit -->
**Served rate** — the rate the API serves — derived from the duration head's enrolment window, band inverted from the duration band:

| Phase | served rate MAE (patients per site per month) | baseline MAE | skill | coverage (nominal) | gate |
|-------|------------------------------------|--------------|-------|--------------------|------|
| P1HV | 21.04 | 23.22 | +0.09 | 0.85 (0.85) | pass |
| P1 | 11.73 | 14.38 | +0.18 | 0.84 (0.80) | pass |
| P2 | 3.68 | 5.37 | +0.32 | 0.82 (0.80) | pass |
| P3 | 10.58 | 15.10 | +0.30 | 0.81 (0.80) | pass |

Fold: horizon fold — train on trials starting before 2018, test on 2018–2020 starts, which have had 5.4–8.6 years to finish against a corpus whose p95 duration is 5.9. Rows P1HV=362, P1=366, P2=367, P3=368 of `experiments/ledger.jsonl`. Regenerate: `python -m experiments.publish_metrics`.

**Cross-check** — the standalone rate head, served only as recruitment_rate_crosscheck (a point, no band):

| Phase | rate-head MAE (patients per site per month) | baseline MAE | skill | coverage (nominal) | gate |
|-------|------------------------------------|--------------|-------|--------------------|------|
| P1HV | 21.06 | 23.22 | +0.09 | 0.80 (0.85) | pass |
| P1 | 11.70 | 14.38 | +0.19 | 0.77 (0.80) | pass |
| P2 | 3.76 | 5.37 | +0.30 | 0.78 (0.80) | pass |
| P3 | 10.52 | 15.10 | +0.30 | 0.76 (0.80) | pass |

Fold: horizon fold — train on trials starting before 2018, test on 2018–2020 starts, which have had 5.4–8.6 years to finish against a corpus whose p95 duration is 5.9. Rows P1HV=352, P1=346, P2=347, P3=348 of `experiments/ledger.jsonl`. Regenerate: `python -m experiments.publish_metrics`.
<!-- published_metrics_rate:end -->

### Known limitations

* **No observed per-site enrolment exists in the public data.** Site-level output
  is modelled from trial-level rates; facility figures are association, not
  attribution.
* **The recruitment rate is approximate.** Its denominator is the full
  start-to-primary-completion span because no enrolment-completion date is
  published, so trials with long follow-up have their rate understated.
* **The corpus is completed trials only**, so recent history is survivorship-
  biased toward fast trials — a trial that started recently and already finished
  is disproportionately a quick one.

---

## Deployment

The app is a single `uvicorn` process — no separate frontend server needed.
Suitable for Railway, Render, or any Docker-capable host.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Author

Katie Lui · [Portfolio](https://klui.bolt.host) · [LinkedIn](http://www.linkedin.com/in/katieluikakiu)
