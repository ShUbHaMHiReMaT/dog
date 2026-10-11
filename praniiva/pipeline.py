"""
Production pipeline: train the bake-off winners, calibrate them, persist everything the dashboard
and a future hardware deployment need, and score any dog (simulated or real).

    measured daily rows  ->  build_features  ->  per-condition model (winner)  ->  calibrated risk %
                                                                              ->  alert (calibrated threshold)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .bakeoff_risk import RISK_CANDIDATES, CoxRisk, GroupStack, evaluate_risk_model, rule_score, time_to_next_event
from .evaluation import calibrate_threshold, simulate_alerts, REFRACTORY
from .features import add_labels, build_features, feature_columns
from .params import CONDITIONS

GATE_PRECISION = 0.30      # >= 30% of alerts must be real
GATE_SENSITIVITY = 0.30    # and >= 30% of events must be caught early
MAKERS = {**RISK_CANDIDATES, "Stacked ensemble (LR+XGB+HGB -> LR)": GroupStack}


def _kind(name):
    if name == "Deterministic rules":
        return "rule"
    if name.startswith("Stacked"):
        return "stack"
    if name.startswith("Survival"):
        return "cox"
    return "clf"


def train_production(F, events, winners: dict, out_dir: Path, neg_frac=0.2, seed=0, verbose=True):
    """For each condition: out-of-fold scores (dogs held out) -> threshold + isotonic calibration;
    then refit on all dogs. Returns OOF risk table for the dashboard."""
    rng = np.random.default_rng(seed)
    feats = feature_columns(F)
    oof_tables = []
    bundle = dict(features=feats, conditions={})
    for cond, name in winners.items():
        kind = _kind(name)
        maker = (lambda H=CONDITIONS[cond]["horizon"]: CoxRisk(H)) if kind == "cox" else MAKERS.get(name)
        metrics, Fe, oof = evaluate_risk_model(F, feats, cond, events, maker, kind, return_scores=True)
        ok = np.isfinite(oof)
        y = Fe[f"y_{cond}"].values
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[ok], y[ok])
        thr = calibrate_threshold(Fe[ok], oof[ok], cond, events)
        em = simulate_alerts(Fe[ok], oof[ok], thr, cond, events)
        # deployment gate: only send alerts for conditions that proved useful on unseen dogs
        deploy = bool(np.nan_to_num(em["alert_precision"]) >= GATE_PRECISION and
                      np.nan_to_num(em["sensitivity"]) >= GATE_SENSITIVITY)
        # final model on all eligible dogs
        X = Fe[feats].values.astype(np.float32)
        keep = (y == 1) | (rng.random(len(y)) < neg_frac)
        if kind == "rule":
            model = None
        elif kind == "cox":
            model = maker().fit(X[keep], y[keep], time_to_next_event(Fe, events, cond)[keep])
        elif kind == "stack":
            model = maker().fit(X[keep], y[keep], Fe.dog_id.values[keep])
        else:
            model = maker().fit(X[keep], y[keep])
        bundle["conditions"][cond] = dict(model_name=name, kind=kind, model=model, threshold=thr, deploy_alerts=deploy,
                                          isotonic=iso, oof_metrics=metrics, oof_alert_metrics=em)
        t = Fe[["dog_id", "day"]].copy()
        t[f"score_{cond}"] = oof
        t[f"risk_{cond}"] = iso.predict(np.nan_to_num(oof))
        t[f"alert_{cond}"] = np.isfinite(oof) & (oof >= thr) & deploy
        oof_tables.append(t)
        if verbose:
            print(f"    {cond:<15} {name:<36} OOF AUROC {metrics['auroc']:.3f}  thr {thr:.3f}  "
                  f"sens {em['sensitivity']:.2f} prec {em['alert_precision']:.2f} FA/dog-mo {em['false_alerts_per_dog_month']:.2f}"
                  f"  -> {'ALERTS ON' if deploy else 'MONITOR ONLY (failed gate)'}")
    joblib.dump(bundle, out_dir / "risk_models.joblib")
    oof = F[["dog_id", "day"]].copy()
    for t in oof_tables:
        oof = oof.merge(t, on=["dog_id", "day"], how="left")
    return bundle, oof


def score(F: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score any feature table (e.g. one new dog) with the production bundle."""
    out = F[["dog_id", "day"]].copy()
    X = F.reindex(columns=bundle["features"]).values.astype(np.float32)
    for cond, b in bundle["conditions"].items():
        if b["kind"] == "rule":
            s = rule_score(F, cond)
        else:
            s = b["model"].predict_proba(X)[:, 1]
        if cond == "chf_decomp" and "mmvd_dx" in F:
            s = np.where(F["mmvd_dx"].values == 1, s, np.nan)       # only for diagnosed cardiac patients
        out[f"score_{cond}"] = s
        out[f"risk_{cond}"] = b["isotonic"].predict(np.nan_to_num(s))
        out[f"risk_{cond}"] = np.where(np.isfinite(s), out[f"risk_{cond}"], np.nan)
        out[f"alert_{cond}"] = np.isfinite(s) & (s >= b["threshold"]) & b.get("deploy_alerts", True)
    return out


def debounce_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the refractory period per dog (what an owner/vet would actually receive)."""
    df = df.sort_values(["dog_id", "day"]).copy()
    for cond in CONDITIONS:
        col = f"alert_{cond}"
        if col not in df:
            continue
        refr = REFRACTORY.get(cond, 7)
        res = np.zeros(len(df), bool)
        on = df[col].fillna(False).astype(bool).values      # unscored days (NaN) are NOT alerts
        for _, idx in df.groupby("dog_id", sort=False).indices.items():
            last = -10_000
            for i in idx:
                if on[i] and df.day.values[i] - last >= refr:
                    res[i] = True; last = df.day.values[i]
        df[f"fired_{cond}"] = res
    return df


def make_custom_dog(breed: str, age: float, weight: float, mmvd=False, atopic=False, otitis=False,
                    outdoor=False, indoor_ac=False, overweight=False, compliance=0.9, act_mult=1.0, seed=0):
    from .params import BREEDS
    info = BREEDS[breed]
    return dict(dog_id="LIVE", breed=breed, group=info["group"], weight=float(weight), age0=float(age), sex="M",
                neutered=True, coat=info["coat"], double_coat=info["double"], cfr=info["cfr"], brachy=info["cfr"] < 0.2,
                coat_colour="mixed", overweight=overweight, indoor_ac=indoor_ac, outdoor=outdoor, act_mult=act_mult,
                compliance=compliance, mmvd=mmvd, atopic=atopic, otitis_prone=otitis, anxiety=0.3)


def simulate_single(dog: dict, inject: dict, config: str, model_dir: Path, bundle: dict | None = None,
                    days: int = 400, seed: int = 0, faults: bool = True):
    """Simulate one dog with injected conditions, measure it, featurise it and score it."""
    from .daily import make_weather, measure, simulate_dog
    wx = make_weather(days, seed=0)
    co = pd.DataFrame([dog])
    rng = np.random.default_rng(seed)
    tr, evs, eps = simulate_dog(co.iloc[0], wx, rng, inject=inject)
    tr = tr.merge(wx.reset_index().rename(columns={"index": "day"}), on="day", how="left")
    events = pd.DataFrame([dict(dog_id=dog["dog_id"], condition=c, event_day=e, onset_day=o) for c, e, o in evs])
    episodes = pd.DataFrame([dict(dog_id=dog["dog_id"], kind=k, start_day=s, length=L) for k, s, L in eps])
    from .quality import assess, clean, maintenance_messages
    from .reference import explain
    meas_q = assess(measure(tr, co, config, model_dir, seed=seed, faults=faults))
    meas = clean(meas_q)
    F = build_features(meas, co, events if len(events) else None)
    bundle = bundle or joblib.load(model_dir / "risk_models.joblib")
    sc = debounce_alerts(score(F, bundle))
    why = []
    for cond in bundle["conditions"]:
        for i in np.where(sc[f"fired_{cond}"].values)[0]:
            why.append(dict(day=int(sc.day.values[i]), condition=cond, risk=float(sc[f"risk_{cond}"].values[i]),
                            why="; ".join(explain(F.iloc[i], cond))))
    return dict(truth=tr, meas=meas_q, F=F, scores=sc, events=events, episodes=episodes,
                maintenance=maintenance_messages(meas_q), alerts=pd.DataFrame(why))
