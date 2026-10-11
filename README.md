# Virtual Patient Digital Twin
**Happiest Health Digital Twin Challenge: Build. Predict. Reimagine Healthcare.**

A proof of concept that fuses **EHR** and **wearable** data into a *virtual patient* that
1. **predicts** 90-day cardiometabolic event risk,
2. **explains** the top modifiable drivers of that risk,
3. **simulates** what-if lifestyle and treatment plans and forecasts the 12-week risk trajectory.

> All data is synthetic. Decision support only, not medical advice.

## Architecture
```
EHR snapshot ─┐                       ┌─ risk score + band
              ├─> feature builder ─> calibrated GBM ─> PatientTwin ─┼─ driver attribution
Wearables 90d ┘  (14d means, trends)                  (state)       └─ what-if simulation + forecast
                                                                          │
                                                                    Streamlit dashboard
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.data_gen          # writes data/ehr.csv, data/wearables.csv
python -m pytest -q tests       # smoke test
streamlit run app.py
```

## Repo layout
- `src/data_gen.py` synthetic EHR + wearables
- `src/model.py` features, training, metrics
- `src/twin.py` `PatientTwin` (state, risk, drivers, simulate, forecast)
- `app.py` dashboard
- `docs/ASSUMPTIONS.md` assumptions, limitations, ethics

## Results (held-out 25%)
AUROC ~0.86, Brier ~0.058 (event rate 8%). Synthetic data, so treat as a pipeline check, not clinical validity.

## Roadmap
FHIR ingestion, real wearable APIs (Apple Health / Fitbit), temporal twin (state-space/LSTM),
SHAP explanations, fairness audit, clinician-in-the-loop review.

## Team
<add team name and members>
