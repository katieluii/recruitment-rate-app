# Model Artifacts

Trained model files live here. This directory is **not committed to git** (see `.gitignore`).

## Directory structure after training

```
models/artifacts/
  P1HV/
    model.pkl       — trained LightGBM pipeline (preprocessor + model)
    metadata.json   — {"rmse": 45.3, "n_train": 1234}
    analytics.json  — pre-computed box-plot stats per therapeutic area
  P1/
    ...
  P2/
    ...
  P3/
    ...
```

## How to generate

```bash
# From the project root, with dependencies installed:
python -m scripts.train_models

# Or train a single phase:
python -m scripts.train_models --phase P2
```

Training fetches ~5 000 completed interventional trials per phase
from the ClinicalTrials.gov API (no account required).
Each phase takes 2–5 minutes depending on API response time.

## Placing pre-trained artifacts

If you have pre-trained `.pkl` files, copy them into the appropriate
subdirectory and restart the server. The app will auto-detect them at startup.
