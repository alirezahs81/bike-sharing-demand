# Bike Sharing Demand Prediction

[![CI/CD Pipeline](https://github.com/alirezahs81/bike-sharing-demand/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/bike-sharing-demand/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predict **hourly bike rental demand** for a Washington, D.C. bike sharing system using the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (17,379 hourly records, 2011–2012). The trained model is served through a FastAPI endpoint.

## Problem

Given calendar and weather features for one hour (season, hour of day, weather situation, temperature, humidity, wind speed, ...), predict how many bikes will be rented (`cnt`). This helps operators plan rebalancing and staffing.

## Key design decisions

- **Leakage-free target**: the raw dataset's `casual` + `registered` columns sum to `cnt`, so both are excluded from features.
- **Chronological split**: data is split by time (80/10/10 train/valid/test), never shuffled, so the test set is true future demand.
- **Cyclical encoding**: hour, weekday and month are encoded with sine/cosine so 23:00 and 00:00 are numerically close.
- **Extra features**: rush-hour flag and a temperature × humidity interaction.

## Results

| Model | Valid R² | Valid MAE | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|---|
| Linear Regression | 0.5651 | 118.6 | | | |
| Random Forest | 0.9170 | 43.4 | | | |
| **Gradient Boosting** (selected) | **0.9303** | **41.1** | **0.8593** | **46.6** | **70.5** |

Top features by importance: hour of day (`hr`), cyclical hour encoding, `workingday`, feels-like temperature, year, rush-hour flag — demand is driven by commute patterns and weather, as expected. Full comparison in `reports/model_comparison.csv`.

## Project structure

```
bike-sharing-demand/
├── src/
│   ├── preprocess.py     # feature engineering + chronological split
│   ├── train.py          # 3 models, selection by validation R²
│   └── api.py            # FastAPI service
├── scripts/
│   └── download_data.py  # fetch UCI dataset
├── tests/                # unit tests (pytest)
├── data/                 # raw + processed (git-ignored, re-downloadable)
├── models/               # saved model artifacts
├── reports/              # metrics CSV
└── .github/workflows/    # CI: lint → train → test
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download + preprocess + train
python scripts/download_data.py
python src/preprocess.py
python src/train.py

# 3. Serve the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"season":1,"yr":1,"mnth":1,"hr":8,"holiday":0,"weekday":3,"workingday":1,"weathersit":1,"temp":0.24,"atemp":0.2879,"hum":0.81,"windspeed":0.0}'
```

Interactive docs: `http://localhost:8000/docs`

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src
```

## License

MIT — see [LICENSE](LICENSE).
