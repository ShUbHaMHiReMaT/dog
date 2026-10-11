"""
Bake-off Parts B and C (see bakeoff.py for Part A).

  Part B  per-dog abnormality detectors on one channel each: rolling-median z (baseline),
          CUSUM change-point, linear-regression trend, ARIMA surprise, Chronos-Bolt surprise,
          and the deterministic RR>30 rule for CHF.
  Part C  multi-feature risk prediction: deterministic rules, logistic regression, HistGB, XGBoost,
          Random Forest, Bayesian (naive-Bayes network), grouped stacked ensemble, Cox survival.
          Each fold: fit on 75% of training dogs, CALIBRATE the alert threshold on the other 25%,
          report event-level precision / recall on held-out test dogs.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .evaluation import calibrate_threshold, day_metrics, simulate_alerts
from .params import CONDITIONS

warnings.filterwarnings("ignore")

# ======================================================================================
# Part B - per-dog abnormality detectors
# ======================================================================================
DETECTOR_TASKS = {           # condition: (channel, direction of change)
    "chf_decomp": ("sleep_rr", +1),
    "pruritic_flare": ("scratch_min", +1),
    "oa_mobility": ("active_min", -1),
    "ckd_pupd": ("drink_min", +1),
    "acute_gi": ("eat_min", -1),
    "tick_fever": ("active_min", -1),
}


def trend_z(meas, c, window=14):
    """Linear-regression slope over the last `window` days, scaled by the dog's robust spread."""
    out = np.full(len(meas), np.nan)
    tt = np.arange(window) - (window - 1) / 2
    for _, idx in meas.groupby("dog_id", sort=False).indices.items():
        y = meas[c].values[idx]
        s = pd.Series(y)
        mad = ((s - s.rolling(28, min_periods=10).median()).abs().rolling(28, min_periods=10).median()
               .values * 1.4826 + 1e-3)
        for k in range(window, len(y)):
            w = y[k - window + 1:k + 1]
            ok = np.isfinite(w)
            if ok.sum() >= window // 2:
                out[idx[k]] = np.polyfit(tt[ok], w[ok], 1)[0] * window / mad[k]
    return out


def bakeoff_detectors(F, meas, events, out_dir: Path, verbose=True, n_dogs_arima=160, seed=0):
    from .forecasters import arima_surprise, chronos_surprise
    rng = np.random.default_rng(seed)
    chans = sorted({c for c, _ in DETECTOR_TASKS.values()})
    mmvd_dogs = F.loc[F.mmvd_dx == 1, "dog_id"].unique()
    other = rng.choice(np.setdiff1d(F.dog_id.unique(), mmvd_dogs), n_dogs_arima, replace=False)
    keep = np.r_[mmvd_dogs, other]
    sub = meas[meas.dog_id.isin(keep)].sort_values(["dog_id", "day"]).reset_index(drop=True)
    if verbose:
        print(f"    Chronos-Bolt + ARIMA on {len(keep)} dogs ...", flush=True)
    t0 = time.time(); ch = chronos_surprise(sub, chans); t_ch = time.time() - t0
    t0 = time.time(); ar = arima_surprise(sub, chans); t_ar = time.time() - t0
    G = (F[F.dog_id.isin(keep)].merge(ch, on=["dog_id", "day"], how="left")
         .merge(ar, on=["dog_id", "day"], how="left").sort_values(["dog_id", "day"]).reset_index(drop=True))
    trends = {c: trend_z(sub, c) for c in chans}
    rows = []
    for cond, (c, sgn) in DETECTOR_TASKS.items():
        dets = {
            "Rolling-median z (baseline)": sgn * G[f"{c}__z3"].values,
            "CUSUM change-point": (G[f"{c}__cusum_up"] if sgn > 0 else G[f"{c}__cusum_dn"]).values,
            "Linear-regression trend": sgn * trends[c],
            "ARIMA surprise": sgn * G[f"{c}__arima_z3"].values,
            "Chronos-Bolt surprise (HF)": sgn * G[f"{c}__chronos_z3"].values,
        }
        if cond == "chf_decomp":
            dets["Rule: sleeping RR > 30 (Rishniw)"] = np.nan_to_num(G["sleep_rr__m3"].values) / 30
        ok = G[f"eval_{cond}"].values
        dogs = G.loc[ok, "dog_id"].unique()
        half = rng.permutation(dogs)[: len(dogs) // 2]
        val = ok & G.dog_id.isin(half).values
        te = ok & ~G.dog_id.isin(half).values
        for name, s in dets.items():
            s = s.astype(float)
            dm = day_metrics(G[f"y_{cond}"].values[ok], s[ok])
            thr = calibrate_threshold(G[val], s[val], cond, events)
            em = simulate_alerts(G[te], s[te], thr, cond, events)
            rows.append(dict(condition=cond, channel=c, detector=name, **dm, **em))
            if verbose:
                print(f"    {cond:<15} {name:<34} AUROC {dm['auroc']:.3f}  sens {em['sensitivity']:.2f}  "
                      f"prec {em['alert_precision']:.2f}  FA/dog-mo {em['false_alerts_per_dog_month']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df["compute_s"] = df.detector.map({"Chronos-Bolt surprise (HF)": round(t_ch, 1), "ARIMA surprise": round(t_ar, 1)})
    df.to_csv(out_dir / "bakeoff_detectors.csv", index=False)
    return df


# ======================================================================================
# Part C - risk prediction
# ======================================================================================
def _imp():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.6, tree_method="hist", n_jobs=-1, verbosity=0)


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def make_lr():
    return make_pipeline(_imp(), StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))


RISK_CANDIDATES = {
    "Logistic regression (linear)": make_lr,
    "HistGB": make_hgb,
    "XGBoost": make_xgb,
    "Random Forest": lambda: make_pipeline(_imp(), RandomForestClassifier(200, min_samples_leaf=20, max_features=0.3,
                                                                           n_jobs=-1, random_state=0)),
    "Bayesian (naive-Bayes network)": lambda: make_pipeline(_imp(), QuantileTransformer(output_distribution="normal",
                                                                                          n_quantiles=200), GaussianNB()),
}


class GroupStack:
    """Stacked ensemble with GROUPED inner folds (meta-features never come from the same dog)."""

    def __init__(self, bases=None, k=3):
        self.bases = bases or {"lr": make_lr, "xgb": make_xgb, "hgb": make_hgb}
        self.k = k

    def fit(self, X, y, groups):
        oof = np.zeros((len(y), len(self.bases)))
        for tr, te in GroupKFold(self.k).split(X, y, groups):
            for j, mk in enumerate(self.bases.values()):
                oof[te, j] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        self.meta = LogisticRegression(max_iter=2000).fit(oof, y)
        self.fitted = [mk().fit(X, y) for mk in self.bases.values()]
        return self

    def predict_proba(self, X):
        return self.meta.predict_proba(np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted]))


class CoxRisk:
    """Survival analysis: Cox PH on landmark rows (time to next event, censored at 2H)."""

    def __init__(self, horizon, max_feats=40):
        self.H, self.k = horizon, max_feats

    def fit(self, X, y, t_event):
        from lifelines import CoxPHFitter
        self.imp = _imp().fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        Xs = self.sc.transform(self.imp.transform(X))
        r = np.array([abs(np.corrcoef(Xs[:, j], y)[0, 1]) if Xs[:, j].std() > 0 else 0 for j in range(Xs.shape[1])])
        self.cols = np.argsort(-np.nan_to_num(r))[: self.k]
        dur = np.clip(np.where(np.isfinite(t_event), t_event, 2 * self.H), 0.5, 2 * self.H)
        ev = (np.isfinite(t_event) & (t_event <= 2 * self.H)).astype(int)
        d = pd.DataFrame(Xs[:, self.cols], columns=[f"f{j}" for j in self.cols])
        d["T"], d["E"] = dur, ev
        self.cph = CoxPHFitter(penalizer=0.5).fit(d, "T", "E")
        return self

    def predict_proba(self, X):
        Xs = self.sc.transform(self.imp.transform(X))
        h = self.cph.predict_partial_hazard(pd.DataFrame(Xs[:, self.cols], columns=[f"f{j}" for j in self.cols])).values
        p = h / (1 + h)
        return np.column_stack([1 - p, p])


def rule_score(F, cond):
    """Deterministic clinician-style rules (no learning). Graded score; >=1 means 'rule fires'."""
    def g(c):
        return np.nan_to_num(F[c].values) if c in F else np.zeros(len(F))
    if cond == "chf_decomp":
        return g("sleep_rr__m3") / 30.0
    if cond == "pruritic_flare":
        return np.maximum(g("scratch_min__z7"), g("headshake_count__z7")) / 3
    if cond == "oa_mobility":
        return (-g("high_int_min__zlong") - g("standing_min__zlong") + 10 * (g("gait_asym__m7") - 0.08)) / 4
    if cond == "acute_gi":
        return -g("eat_min__z3") / 2.5
    if cond == "tick_fever":
        return (-g("active_min__z3") - g("eat_min__z3")) / 4
    if cond == "ckd_pupd":
        return g("drink_min__zlong") / 2.5
    raise KeyError(cond)


def time_to_next_event(F, events, cond):
    t = np.full(len(F), np.inf)
    for dog_id, grp in events[events.condition == cond].groupby("dog_id"):
        idx = np.where(F.dog_id.values == dog_id)[0]
        e = np.sort(grp.event_day.values)
        days = F.day.values[idx]
        k = np.searchsorted(e, days, side="right")
        t[idx] = np.where(k < len(e), e[np.minimum(k, len(e) - 1)] - days, np.inf)
    return t


def evaluate_risk_model(F, feats, cond, events, make, kind="clf", n_folds=3, neg_frac=0.2, seed=0,
                        return_scores=False, split="dog", fixed_threshold=None):
    """Outer CV grouped by dog (split='dog') or naive row split (split='random', to show inflation)."""
    rng = np.random.default_rng(seed)
    Fe = F[F[f"eval_{cond}"].values].reset_index(drop=True)
    X = Fe[feats].values.astype(np.float32)
    y = Fe[f"y_{cond}"].values
    grp = Fe.dog_id.values
    tte = time_to_next_event(Fe, events, cond) if kind == "cox" else None
    if split == "random":
        fold_of = rng.integers(0, n_folds, len(y))
        outer = [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_folds)]
    else:
        outer = list(GroupKFold(n_folds).split(X, y, grp))
    scores = np.full(len(y), np.nan)
    ems = []
    for tr, te in outer:
        tdogs = np.unique(grp[tr])
        vdogs = rng.permutation(tdogs)[: max(1, len(tdogs) // 4)]
        is_val = np.isin(grp[tr], vdogs)
        fit_idx, val_idx = tr[~is_val], tr[is_val]
        fi = fit_idx[(y[fit_idx] == 1) | (rng.random(len(fit_idx)) < neg_frac)]
        if kind == "rule":
            s_val, s_te = rule_score(Fe.iloc[val_idx], cond), rule_score(Fe.iloc[te], cond)
        else:
            if y[fi].sum() < 3:
                continue
            if kind == "cox":
                m = make().fit(X[fi], y[fi], tte[fi])
            elif kind == "stack":
                m = make().fit(X[fi], y[fi], grp[fi])
            else:
                m = make().fit(X[fi], y[fi])
            s_val, s_te = m.predict_proba(X[val_idx])[:, 1], m.predict_proba(X[te])[:, 1]
        thr = fixed_threshold if fixed_threshold is not None else calibrate_threshold(Fe.iloc[val_idx], s_val, cond, events)
        if split == "dog":
            ems.append(simulate_alerts(Fe.iloc[te], s_te, thr, cond, events))
        scores[te] = s_te
    out = dict(**day_metrics(y, scores))
    if ems:
        agg = pd.DataFrame(ems)
        w = agg.events.clip(lower=1)
        out.update(events=int(agg.events.sum()),
                   sensitivity=float(np.average(agg.sensitivity.fillna(0), weights=w)),
                   alert_precision=float(np.nanmean(agg.alert_precision)) if agg.alert_precision.notna().any() else np.nan,
                   false_alerts_per_dog_month=float(agg.false_alerts_per_dog_month.mean()),
                   median_lead_days=float(np.nanmedian(agg.median_lead_days)) if agg.median_lead_days.notna().any() else np.nan)
    if return_scores:
        return out, Fe, scores
    return out


def bakeoff_risk(F, feats, events, out_dir: Path, conditions=None, verbose=True, include=None):
    rows = []
    for cond in conditions or CONDITIONS:
        H = CONDITIONS[cond]["horizon"]
        cands = {"Deterministic rules": (None, "rule"),
                 **{k: (v, "clf") for k, v in RISK_CANDIDATES.items()},
                 "Stacked ensemble (LR+XGB+HGB -> LR)": (GroupStack, "stack"),
                 "Survival analysis (Cox PH)": (lambda H=H: CoxRisk(H), "cox")}
        if include:
            cands = {k: v for k, v in cands.items() if k in include}
        for name, (mk, kind) in cands.items():
            t0 = time.time()
            r = evaluate_risk_model(F, feats, cond, events, mk, kind)
            rows.append(dict(condition=cond, model=name, seconds=round(time.time() - t0, 1), **r))
            if verbose:
                print(f"    {cond:<15} {name:<38} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f} (prev {r['prevalence']:.3f})"
                      f" | sens {r['sensitivity']:.2f} prec {r['alert_precision']:.2f} "
                      f"FA/dog-mo {r['false_alerts_per_dog_month']:.2f} lead {r['median_lead_days']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bakeoff_risk.csv", index=False)
    return df
