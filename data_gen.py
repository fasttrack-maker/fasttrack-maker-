"""Synthetic EHR + wearable data generator.

All data is synthetic. No real patient information is used.
Outcome: 90-day adverse cardiometabolic event (e.g. admission for
hypertensive crisis / diabetic decompensation), driven by a latent risk
function of EHR baseline + recent wearable behaviour.
"""
import numpy as np
import pandas as pd

DAYS = 90


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n_patients: int = 1500, seed: int = 42):
    rng = np.random.default_rng(seed)
    ehr_rows, wear_rows = [], []
    for pid in range(n_patients):
        age = int(np.clip(rng.normal(52, 13), 25, 85))
        sex = rng.choice(["F", "M"])
        bmi = float(np.clip(rng.normal(27.5, 4.5), 17, 45))
        diabetic = rng.random() < _sigmoid((bmi - 28) / 3 + (age - 50) / 25 - 1.2)
        hba1c = float(np.clip(rng.normal(7.4 if diabetic else 5.5, 0.9 if diabetic else 0.4), 4.5, 12))
        sbp = float(np.clip(rng.normal(118 + (age - 50) * 0.5 + (bmi - 27) * 0.8, 12), 90, 190))
        ldl = float(np.clip(rng.normal(115 + (bmi - 27) * 1.5, 28), 50, 220))
        smoker = bool(rng.random() < 0.18)
        on_meds = bool((sbp > 135 or diabetic) and rng.random() < 0.75)
        adherence = float(np.clip(rng.beta(5, 2), 0.2, 1.0)) if on_meds else 1.0

        fit = rng.normal(0, 1)
        base_steps = np.clip(7000 + 2500 * fit - (age - 50) * 40, 1200, 16000)
        base_sleep = np.clip(rng.normal(6.8, 0.7), 4.5, 9)
        base_rhr = np.clip(68 - 4 * fit + (bmi - 27) * 0.5 + rng.normal(0, 3), 48, 95)
        base_hrv = np.clip(48 + 10 * fit - (age - 50) * 0.5 + rng.normal(0, 5), 15, 110)
        drift = rng.normal(0, 1)
        t = np.arange(DAYS)
        steps = np.clip(base_steps * (1 + 0.15 * drift * t / DAYS) + rng.normal(0, 1200, DAYS), 300, 25000)
        sleep = np.clip(base_sleep - 0.4 * max(-drift, 0) * t / DAYS + rng.normal(0, 0.7, DAYS), 3, 10)
        rhr = np.clip(base_rhr + 2.5 * (-drift) * t / DAYS + rng.normal(0, 2, DAYS), 42, 110)
        hrv = np.clip(base_hrv - 5 * (-drift) * t / DAYS + rng.normal(0, 5, DAYS), 10, 120)
        spo2 = np.clip(rng.normal(97 - (0.8 if smoker else 0), 1.0, DAYS), 88, 100)

        recent = slice(DAYS - 14, DAYS)
        z = (
            -4.6
            + 0.025 * (age - 50)
            + 0.9 * (hba1c - 5.5) * (1 if diabetic else 0.4)
            + 0.025 * (sbp - 120)
            + 0.006 * (ldl - 110)
            + 0.5 * smoker
            + 0.06 * (bmi - 27)
            - 0.00028 * (steps[recent].mean() - 7000)
            - 0.35 * (sleep[recent].mean() - 6.8)
            + 0.06 * (rhr[recent].mean() - 68)
            - 0.025 * (hrv[recent].mean() - 45)
            + 1.4 * (1 - adherence) * on_meds
            + rng.normal(0, 0.35)
        )
        event = int(rng.random() < _sigmoid(z))

        ehr_rows.append(dict(patient_id=pid, age=age, sex=sex, bmi=round(bmi, 1),
                             diabetic=int(diabetic), hba1c=round(hba1c, 1), sbp=round(sbp),
                             ldl=round(ldl), smoker=int(smoker), on_meds=int(on_meds),
                             med_adherence=round(adherence, 2), event_90d=event))
        for d in range(DAYS):
            wear_rows.append(dict(patient_id=pid, day=d, steps=int(steps[d]), sleep_hrs=round(sleep[d], 2),
                                  resting_hr=round(rhr[d], 1), hrv_ms=round(hrv[d], 1), spo2=round(spo2[d], 1)))
    return pd.DataFrame(ehr_rows), pd.DataFrame(wear_rows)


if __name__ == "__main__":
    ehr, wear = generate()
    ehr.to_csv("data/ehr.csv", index=False)
    wear.to_csv("data/wearables.csv", index=False)
    print(f"EHR {ehr.shape}, wearables {wear.shape}, event rate {ehr.event_90d.mean():.1%}")
