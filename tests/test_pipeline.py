"""Safety rules of the alerting pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from praniiva.params import CONDITIONS
from praniiva.pipeline import bootstrap_ci, debounce_alerts, per_dog_alert_stats, score


class Const:
    def __init__(self, p): self.p = p
    def predict_proba(self, X): return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])


def fake_bundle(features):
    iso = IsotonicRegression(out_of_bounds="clip").fit([0, 1], [0, 1])
    conds = {c: dict(kind="clf", model=Const(0.9), threshold=0.5, isotonic=iso, deploy_alerts=True) for c in CONDITIONS}
    return dict(features=features, conditions=conds)


def frame(n=10, mmvd=1):
    F = pd.DataFrame(dict(dog_id="A", day=np.arange(n), mmvd_dx=mmvd))
    for ch in ("high_int_min", "scratch_min", "sleep_rr", "eat_min", "active_min", "drink_min"):
        F[f"{ch}__z7"] = 0.0
    return F


def test_no_score_without_key_measurement():
    F = frame()
    F.loc[3:5, "scratch_min__z7"] = np.nan          # e.g. collar loose -> scratching masked
    out = score(F, fake_bundle(list(F.columns[3:])))
    assert out.loc[3:5, "score_pruritic_flare"].isna().all()
    assert (out.loc[3:5, "status_pruritic_flare"] == "insufficient_data").all()
    assert not out.loc[3:5, "alert_pruritic_flare"].any()
    assert (out.loc[0:2, "status_pruritic_flare"] == "ok").all()


def test_missing_trained_features_are_flagged():
    F = frame()
    out = score(F, fake_bundle(list(F.columns[3:]) + ["skin_temp__z7"]))   # model trained with a sensor we lack
    assert (out["status_oa_mobility"] == "reduced_inputs").all()
    assert out.attrs["missing_features"] == ["skin_temp__z7"]


def test_chf_only_for_diagnosed_cardiac_dogs():
    out = score(frame(mmvd=0), fake_bundle(list(frame().columns[3:])))
    assert out["score_chf_decomp"].isna().all()
    assert (out["status_chf_decomp"] == "not_applicable").all()


def test_concern_is_score_over_threshold():
    out = score(frame(), fake_bundle(list(frame().columns[3:])))
    assert np.allclose(out["concern_oa_mobility"], 0.9 / 0.5)


def test_nan_is_never_an_alert():
    df = pd.DataFrame(dict(dog_id="A", day=np.arange(5), alert_oa_mobility=[np.nan, None, np.nan, False, np.nan]))
    assert not debounce_alerts(df)["fired_oa_mobility"].any()


def test_no_alerts_after_diagnosis():
    days = np.arange(120)
    df = pd.DataFrame(dict(dog_id="A", day=days, alert_oa_mobility=True, alert_acute_gi=True))
    dx = pd.DataFrame([dict(dog_id="A", condition="oa_mobility", event_day=50),
                       dict(dog_id="A", condition="acute_gi", event_day=50)])
    out = debounce_alerts(df, diagnoses=dx)
    oa = out[out.fired_oa_mobility].day.values
    gi = out[out.fired_acute_gi].day.values
    assert oa.max() < 50                                   # chronic: never after diagnosis
    assert not ((gi >= 50) & (gi <= 71)).any()             # acute: 21-day treatment window
    assert (gi > 71).any()                                 # ...then monitoring resumes


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(60):
        ev = rng.integers(0, 2)
        rows.append(dict(dog_id=d, dog_days=300, alerts=rng.integers(0, 4), true_alerts=0, events=ev,
                         caught=ev * rng.integers(0, 2)))
    st = pd.DataFrame(rows)
    st["true_alerts"] = np.minimum(st.alerts, st.caught)
    ci = bootstrap_ci(st, n_boot=200)
    for k in ("sensitivity", "alert_precision", "false_alerts_per_dog_month"):
        assert ci[f"{k}_lo"] <= ci[k] <= ci[f"{k}_hi"]


def test_per_dog_stats_counts_true_alerts():
    df = pd.DataFrame(dict(dog_id="A", day=np.arange(100), eval_chf_decomp=True))
    s = np.zeros(100); s[[40, 90]] = 1.0                   # one alert inside the warning window, one outside
    ev = pd.DataFrame([dict(dog_id="A", condition="chf_decomp", event_day=45, onset_day=35)])
    st = per_dog_alert_stats(df, s, 0.5, "chf_decomp", ev)
    r = st.iloc[0]
    assert (r.alerts, r.true_alerts, r.events, r.caught) == (2, 1, 1, 1)
