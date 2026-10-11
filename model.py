"""Feature engineering + calibrated risk model."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

EHR_FEATURES = ["age", "male", "bmi", "diabetic", "hba1c", "sbp", "ldl", "smoker", "on_meds", "med_adherence"]
WEAR_FEATURES = ["steps_14d", "sleep_14d", "rhr_14d", "hrv_14d", "spo2_14d",
                 "steps_trend", "rhr_trend", "hrv_trend", "sleep_var"]
FEATURES = EHR_FEATURES + WEAR_FEATURES


def wearable_features(w: pd.DataFrame) -> dict:
    """Summarise a patient's daily wearable series."""
    w = w.sort_values("day")
    last, prev = w.tail(14), w.iloc[-28:-14]
    return dict(
        steps_14d=last.steps.mean(), sleep_14d=last.sleep_hrs.mean(),
        rhr_14d=last.resting_hr.mean(), hrv_14d=last.hrv_ms.mean(), spo2_14d=last.spo2.mean(),
        steps_trend=last.steps.mean() - prev.steps.mean(),
        rhr_trend=last.resting_hr.mean() - prev.resting_hr.mean(),
        hrv_trend=last.hrv_ms.mean() - prev.hrv_ms.mean(),
        sleep_var=last.sleep_hrs.std(),
    )


def build_features(ehr: pd.DataFrame, wear: pd.DataFrame) -> pd.DataFrame:
    wf = pd.DataFrame({pid: wearable_features(g) for pid, g in wear.groupby("patient_id")}).T
    wf.index.name = "patient_id"
    df = ehr.merge(wf.reset_index(), on="patient_id")
    df["male"] = (df.sex == "M").astype(int)
    return df


def train(df: pd.DataFrame, seed: int = 0):
    X, y = df[FEATURES], df["event_90d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    base = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    metrics = dict(auroc=roc_auc_score(yte, p), brier=brier_score_loss(yte, p),
                   base_rate=float(y.mean()), n_test=len(yte))
    return model, metrics
