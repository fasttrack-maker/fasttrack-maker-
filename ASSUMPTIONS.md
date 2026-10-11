# Assumptions and Limitations

## Data
- Fully synthetic: 1,500 patients, 90 days of daily wearable data each, plus an EHR snapshot.
- The 90-day outcome is generated from a hand-written risk function (age, HbA1c, BP, LDL, smoking, BMI,
  adherence, recent steps / sleep / resting HR / HRV). The model therefore recovers known structure;
  AUROC on real data will be lower.

## What-if physiology rules (src/twin.py)
| Change | Effect applied |
|---|---|
| +1000 steps/day | resting HR -0.6 bpm, HRV +0.9 ms |
| +1 hr sleep | HRV +2 ms |
| Adherence +0.1 (on meds) | SBP -1.2 mmHg, HbA1c -0.06 (diabetics) |
| -1 kg | SBP -1 mmHg, HbA1c -0.05 (diabetics) |
| Quit smoking | smoker flag set to 0 |
These are illustrative, literature-inspired assumptions, not validated causal estimates.

## Model
- Gradient boosting + isotonic calibration; 75/25 stratified split.
- Driver explanations are one-at-a-time counterfactuals to a healthy reference, not SHAP.

## Known limitations
- Forecast curve can be slightly bumpy because tree models are piecewise constant.
- No temporal model of individual trajectories (a natural next step: state-space / LSTM twin).
- Needs consent, de-identification (e.g. FHIR + HIPAA/DPDP compliance) and bias audit before real use.
