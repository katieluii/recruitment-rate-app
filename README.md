# Clinical Trial Duration Predictor

> A full-stack ML web app that predicts how long a clinical trial will take,
> powered by live data from [ClinicalTrials.gov](https://clinicaltrials.gov)
> and phase-specific LightGBM models.

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
| `POST` | `/api/predict` | Predict trial duration |
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
  "region": "US"
}
```

Response:

```json
{
  "predicted_months": 24.3,
  "lower_months": 18.1,
  "upper_months": 30.5,
  "predicted_days": 740,
  "rmse_days": 95.2,
  "n_train": 3241,
  "model_used": "LightGBM",
  "confidence_pct": 80
}
```

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

| Phase | Model | Features | Target |
|-------|-------|----------|--------|
| P1 HV | LightGBM | Enrollment, site count, TA, region, masking, arms, … | Days to primary completion |
| P1    | LightGBM | + SAD/MAD classification | Days to primary completion |
| P2    | LightGBM | As above | Days to primary completion |
| P3    | LightGBM | As above | Days to primary completion |

Uncertainty is reported as an **80% prediction interval** (±1.28 × RMSE,
normal approximation).

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

Katie Lui · [Portfolio](https://klui.bolt.host) · [LinkedIn](#)
