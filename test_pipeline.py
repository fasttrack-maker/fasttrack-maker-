from src.data_gen import generate
from src.model import build_features, train, FEATURES
from src.twin import PatientTwin


def test_end_to_end():
    ehr, wear = generate(n_patients=600, seed=1)
    df = build_features(ehr, wear)
    model, m = train(df)
    assert m["auroc"] > 0.70
    row = df.sort_values("event_90d").iloc[-1]
    twin = PatientTwin(state=row[FEATURES].astype(float), model=model)
    r = twin.simulate(steps_delta=3000, sleep_delta=1.0, quit_smoking=True)
    assert 0 <= r["scenario"] <= 1
    assert len(twin.forecast(weeks=8, steps_delta=2000)) == 9
