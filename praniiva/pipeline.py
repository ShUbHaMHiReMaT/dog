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

GATE_PRECISION = 0.30      # research filter: >= 30% of alerts must be real ...
GATE_SENSITIVITY = 0.30    # ... and >= 30% of events caught early, on the UNTOUCHED test cohort
GATE_MIN_EVENTS = 10       # fewer test events than this = not enough evidence to switch alerts on
MAKERS = {**RISK_CANDIDATES, "Stacked ensemble (LR+XGB+HGB -> LR)": GroupStack}
PRIMARY_CHANNEL = {"oa_mobility": "high_int_min", "pruritic_flare": "scratch_min", "chf_decomp": "sleep_rr",
                   "acute_gi": "eat_min", "tick_fever": "active_min", "ckd_pupd": "drink_min"}
DIAGNOSED_SUPPRESS_DAYS = 21   # acute conditions: no new alerts while under treatment


def _kind(name):
    if name == "Deterministic rules":
        return "rule"
    if name.startswith("Stacked"):
        return "stack"
    if name.startswith("Survival"):
        return "cox"
    return "clf"


def _raw_scores(F, b, cond, X):
    s = rule_score(F, cond) if b["kind"] == "rule" else b["model"].predict_proba(X)[:, 1]
    if cond == "chf_decomp" and "mmvd_dx" in F:
        s = np.where(F["mmvd_dx"].values == 1, s, np.nan)       # only for diagnosed cardiac patients
    return np.asarray(s, dtype=float)


# --------------------------------------------------------------------------------------
# per-dog alert statistics + dog-level bootstrap confidence intervals
# --------------------------------------------------------------------------------------
def per_dog_alert_stats(df, score_, thr, cond, events):
    from .params import positive_window
    refr = REFRACTORY.get(cond, 7)
    wins = {}
    for _, e in events[events.condition == cond].iterrows():
        wins.setdefault(e.dog_id, []).append(positive_window(cond, e.event_day, e.onset_day))
    rows = []
    w = pd.DataFrame(dict(dog_id=df.dog_id.values, day=df.day.values, s=score_, ok=df[f"eval_{cond}"].values))
    for dog, g in w.groupby("dog_id", sort=False):
        g = g[g.ok]
        if g.empty:
            continue
        alerts, last = [], -10_000
        for day, sc in zip(g.day.values, g.s.values):
            if np.isfinite(sc) and sc >= thr and day - last >= refr:
                alerts.append(day); last = day
        dw = wins.get(dog, [])
        true = sum(any(a0 <= a < b0 for a0, b0 in dw) for a in alerts)
        obs = [(a0, b0) for a0, b0 in dw if b0 > g.day.min() and a0 <= g.day.max()]
        caught = [any(a0 <= a < b0 for a in alerts) for a0, b0 in obs]
        rows.append(dict(dog_id=dog, dog_days=len(g), alerts=len(alerts), true_alerts=true,
                         events=len(obs), caught=sum(caught)))
    return pd.DataFrame(rows)


def _agg(st):
    return dict(sensitivity=st.caught.sum() / st.events.sum() if st.events.sum() else np.nan,
                alert_precision=st.true_alerts.sum() / st.alerts.sum() if st.alerts.sum() else np.nan,
                false_alerts_per_dog_month=(st.alerts.sum() - st.true_alerts.sum()) / max(st.dog_days.sum() / 30, 1e-9))


def bootstrap_ci(st, n_boot=500, seed=0):
    """95% CI by resampling DOGS (the unit of independence), not days."""
    rng = np.random.default_rng(seed)
    point = _agg(st)
    reps = {k: [] for k in point}
    for _ in range(n_boot):
        b = st.iloc[rng.integers(0, len(st), len(st))]
        for k, v in _agg(b).items():
            reps[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(reps[k], float)
        arr = arr[np.isfinite(arr)]
        out[k] = v
        out[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if len(arr) else np.nan
        out[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if len(arr) else np.nan
    out["events"] = int(st.events.sum())
    out["dogs"] = len(st)
    return out


def calibration_report(y, p, bins=10):
    """Does a calibrated value of 0.3 mean 30%? (checked on the untouched test cohort)"""
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0 or y.sum() == 0:
        return dict(brier=np.nan, ece=np.nan)
    order = np.argsort(p)
    chunks = [c for c in np.array_split(order, bins) if len(c)]
    ece = sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in chunks)
    return dict(brier=float(np.mean((p - y) ** 2)), ece=float(ece),
                top_decile_pred=float(p[chunks[-1]].mean()), top_decile_obs=float(y[chunks[-1]].mean()))


# --------------------------------------------------------------------------------------
def train_production(F, events, winners: dict, out_dir: Path, F_test=None, events_test=None,
                     neg_frac=0.2, seed=0, verbose=True):
    """
    1. DEVELOPMENT dogs: grouped out-of-fold scores -> isotonic calibration + alert threshold (both fitted on
       out-of-fold predictions only), then the final model is fitted on all development dogs.
    2. Everything is FROZEN.
    3. TEST dogs (a separate cohort never used for model choice, calibration or thresholds): the frozen model,
       threshold and calibration are evaluated once, with dog-level bootstrap 95% CIs.
    4. Alerts are switched on only if the TEST result passes the gate.
    """
    from .evaluation import day_metrics
    rng = np.random.default_rng(seed)
    feats = feature_columns(F)
    oof_tables, test_rows = [], []
    bundle = dict(features=feats, conditions={})
    for cond, name in winners.items():
        kind = _kind(name)
        maker = (lambda H=CONDITIONS[cond]["horizon"]: CoxRisk(H)) if kind == "cox" else MAKERS.get(name)
        metrics, Fe, oof = evaluate_risk_model(F, feats, cond, events, maker, kind, return_scores=True)
        ok = np.isfinite(oof)
        y = Fe[f"y_{cond}"].values
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[ok], y[ok])
        thr = calibrate_threshold(Fe[ok], oof[ok], cond, events)
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
        b = dict(model_name=name, kind=kind, model=model, threshold=thr, isotonic=iso, oof_metrics=metrics)
        dev_stats = bootstrap_ci(per_dog_alert_stats(Fe[ok], oof[ok], thr, cond, events))
        b["dev_alert_metrics"] = dev_stats
        # ---- frozen evaluation on the untouched test cohort ----
        test = None
        if F_test is not None:
            Ft = F_test[F_test[f"eval_{cond}"].values].reset_index(drop=True)
            st_raw = _raw_scores(Ft, b, cond, Ft.reindex(columns=feats).values.astype(np.float32))
            test = bootstrap_ci(per_dog_alert_stats(Ft, st_raw, thr, cond, events_test))
            test.update(calibration_report(Ft[f"y_{cond}"].values, iso.predict(np.nan_to_num(st_raw))))
            test.update(day_metrics(Ft[f"y_{cond}"].values, st_raw))
            test_rows.append(dict(condition=cond, model=name, threshold=thr, **test))
        basis = test if test is not None else dev_stats
        enough = basis["events"] >= GATE_MIN_EVENTS
        deploy = bool(enough and np.nan_to_num(basis["alert_precision"]) >= GATE_PRECISION
                      and np.nan_to_num(basis["sensitivity"]) >= GATE_SENSITIVITY)
        b["deploy_alerts"] = deploy
        b["gate_basis"] = "untouched test cohort" if test is not None else "development out-of-fold (optimistic)"
        b["test_alert_metrics"] = test
        bundle["conditions"][cond] = b
        t = Fe[["dog_id", "day"]].copy()
        t[f"score_{cond}"] = oof
        t[f"concern_{cond}"] = oof / thr if thr > 0 else oof
        t[f"risk_{cond}"] = iso.predict(np.nan_to_num(oof))
        t[f"alert_{cond}"] = np.isfinite(oof) & (oof >= thr) & deploy
        oof_tables.append(t)
        if verbose:
            def f(d, k):
                v = d.get(k, np.nan)
                return f"{v:.2f} [{d[k + '_lo']:.2f}-{d[k + '_hi']:.2f}]" if np.isfinite(v) else "n/a"
            print(f"    {cond:<15} {name:<30} {'TEST' if test is not None else 'DEV'} "
                  f"({basis['events']} events, {basis['dogs']} dogs): sens {f(basis, 'sensitivity')}  "
                  f"prec {f(basis, 'alert_precision')}  FA/dog-mo {f(basis, 'false_alerts_per_dog_month')}"
                  f"  -> {'ALERTS ON' if deploy else 'MONITOR ONLY' + ('' if enough else ' (too few test events)')}",
                  flush=True)
    joblib.dump(bundle, out_dir / "risk_models.joblib")
    if test_rows:
        pd.DataFrame(test_rows).to_csv(out_dir.parent / "reports" / "final_test.csv", index=False)
    oof = F[["dog_id", "day"]].copy()
    for t in oof_tables:
        oof = oof.merge(t, on=["dog_id", "day"], how="left")
    return bundle, oof


def score(F: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score any feature table. Never returns a score that is not backed by data:
       status_<cond> = ok | insufficient_data (key measurement missing or personal baseline not learned yet)
                       | reduced_inputs (scored, but some trained features are unavailable on this device)
                       | not_applicable (e.g. CHF model for a dog without diagnosed MMVD)."""
    out = F[["dog_id", "day"]].copy()
    missing = [c for c in bundle["features"] if c not in F.columns]
    X = F.reindex(columns=bundle["features"]).values.astype(np.float32)
    for cond, b in bundle["conditions"].items():
        s = _raw_scores(F, b, cond, X)
        key = f"{PRIMARY_CHANNEL[cond]}__z7"
        have = F[key].notna().values if key in F else np.zeros(len(F), bool)
        s = np.where(have, s, np.nan)
        status = np.where(~have, "insufficient_data", "reduced_inputs" if missing else "ok").astype(object)
        if cond == "chf_decomp" and "mmvd_dx" in F:
            status = np.where(F["mmvd_dx"].values == 1, status, "not_applicable")
        thr = b["threshold"]
        out[f"score_{cond}"] = s
        out[f"concern_{cond}"] = s / thr if thr > 0 else s
        out[f"risk_{cond}"] = np.where(np.isfinite(s), b["isotonic"].predict(np.nan_to_num(s)), np.nan)
        out[f"alert_{cond}"] = np.isfinite(s) & (s >= thr) & b.get("deploy_alerts", True)
        out[f"status_{cond}"] = status
    out.attrs["missing_features"] = missing
    return out


def debounce_alerts(df: pd.DataFrame, diagnoses: pd.DataFrame | None = None) -> pd.DataFrame:
    """What an owner/vet actually receives: refractory period per dog, and no new alerts for a condition the
    vet has already diagnosed (chronic: from diagnosis on -> treatment tracking; acute: 21-day treatment window)."""
    df = df.sort_values(["dog_id", "day"]).copy()
    dx = {}
    if diagnoses is not None and len(diagnoses):
        for _, r in diagnoses.iterrows():
            dx.setdefault((r.dog_id, r.condition), []).append(int(r.event_day))
    for cond in CONDITIONS:
        col = f"alert_{cond}"
        if col not in df:
            continue
        refr = REFRACTORY.get(cond, 7)
        chronic = CONDITIONS[cond].get("chronic", False)
        res = np.zeros(len(df), bool)
        on = (df[col] == True).to_numpy()                   # noqa: E712 - unscored days (NaN/None) are NOT alerts
        for dog, idx in df.groupby("dog_id", sort=False).indices.items():
            last = -10_000
            dd = dx.get((dog, cond), [])
            for i in idx:
                day = df.day.values[i]
                if any(day >= d and (chronic or day <= d + DIAGNOSED_SUPPRESS_DAYS) for d in dd):
                    continue
                if on[i] and day - last >= refr:
                    res[i] = True; last = day
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
    sc = debounce_alerts(score(F, bundle), diagnoses=events if len(events) else None)
    why = []
    for cond in bundle["conditions"]:
        for i in np.where(sc[f"fired_{cond}"].values)[0]:
            why.append(dict(day=int(sc.day.values[i]), condition=cond, concern=float(sc[f"concern_{cond}"].values[i]),
                            why="; ".join(explain(F.iloc[i], cond))))
    return dict(truth=tr, meas=meas_q, F=F, scores=sc, events=events, episodes=episodes,
                maintenance=maintenance_messages(meas_q), alerts=pd.DataFrame(why))
