# Virtual Patient Digital Twin
**Happiest Health Digital Twin Challenge: Build. Predict. Reimagine Healthcare.**

## 1. Team Details
| Name | Role | Email |
|---|---|---|
| Sarthak Sonkar | <role> | <email> |
| Supriya Suman | <role> | <email> |
| Sonam Kumari | <role> | <email> |

Team name: CardioSync

## 2. College / Incubator Information

## 3. Project Title
Virtual Patient Digital Twin: Predicting Health Outcomes from EHR and Wearable Data

## 4. Problem Statement
Clinicians see a patient only during visits, while risk builds up between them. EHR snapshots and
wearable streams sit in separate silos, so deterioration is caught late and lifestyle advice is generic.
Challenge: build a virtual patient model that predicts health outcomes using EHR and wearable data.

## 5. Healthcare Use Case
Early warning and preventive care for chronic cardiometabolic patients (diabetes, hypertension):
1. Predict 90-day adverse event risk.
2. Explain the top modifiable drivers.
3. Simulate what-if plans (steps, sleep, adherence, quitting smoking, weight loss) and forecast 12-week risk.
Users: care teams, health coaches and patients.

## 6. Technical Stack
Python 3.10+, pandas, NumPy, scikit-learn, Streamlit.

## 7. AI/ML Model / Framework Details
- Features: EHR baseline (age, sex, BMI, HbA1c, BP, LDL, smoking, medication adherence) plus wearable
  features (14-day means of steps, sleep, resting HR, HRV, SpO2; trends; sleep variability).
- Model: Gradient Boosting Classifier with isotonic probability calibration (scikit-learn).
- Twin engine (`src/twin.py`): holds patient state, predicts risk, explains drivers by counterfactual
  attribution, and simulates interventions with documented physiology rules.
- Held-out result (synthetic data): AUROC ~0.86, Brier ~0.058.
- Data is fully synthetic; limitations in `docs/ASSUMPTIONS.md`.

## 8. Architecture Diagram
See `docs/architecture.pdf` (PDF).
```
EHR snapshot ─┐                       ┌─ risk score + band
              ├─> feature builder ─> calibrated GBM ─> PatientTwin ─┼─ driver attribution
Wearables 90d ┘  (14d means, trends)                  (state)       └─ what-if simulation + forecast
                                                                          │
                                                                    Streamlit dashboard
```

## 9. Presentation
See `` <presentation.pdf>.

## 10. Demo Video
Unlisted YouTube link (): `<[paste link here]([https://www.veed.io/ai-studio/32401f1f-7416-4b8b-b53d-8603bc1171da?source=dashboard](https://youtu.be/k3ityQRLrYo?si=CsOVOY8zn_FNKAzi))>`

## 11. Run Locally
```bash
pip install -r requirements.txt
python -m src.data_gen
python -m pytest -q tests
streamlit run app.py
```

## 12. Open-Source License
MIT License (see `LICENSE`).

## 13. Access
All files and links in this repository are publicly accessible without additional permissions.
