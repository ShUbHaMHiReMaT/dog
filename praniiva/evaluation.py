"""
Evaluation & threshold calibration - the part that decides whether anything is "good".

Metrics reported for every model and condition:
  day-level  : AUROC, AUPRC (prevalence printed alongside - AUPRC must be read against it)
  event-level: sensitivity  (event caught by >=1 alert in the H days before it)
               alert precision (alerts followed by a real event within H days)
               false alerts per dog-month
               median lead time (days between first alert and the event)
Thresholds are chosen on VALIDATION dogs that the model was not trained on, then frozen and
applied to TEST dogs - so the reported precision/recall is not tuned on the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .params import CONDITIONS, positive_window

REFRACTORY = {"oa_mobility": 30, "ckd_pupd": 30}      # days an alert suppresses the next; default 7
MAX_FALSE_PER_DOG_MONTH = 0.25                         # alert budget: ~1 false alert per dog per 4 months


def day_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return dict(auroc=np.nan, auprc=np.nan, prevalence=float(y.mean()) if len(y) else np.nan)
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                prevalence=float(y.mean()))


def simulate_alerts(df: pd.DataFrame, score: np.ndarray, thr: float, cond: str, events: pd.DataFrame):
    """df: rows (dog_id, day, eval_<cond>) aligned with score. Returns event-level metrics."""
    H = CONDITIONS[cond]["horizon"]
    refr = REFRACTORY.get(cond, 7)
    ev = events[events.condition == cond]
    win_by_dog = {d: [positive_window(cond, e, o) for e, o in zip(g.event_day.values, g.onset_day.values)]
                  for d, g in ev.groupby("dog_id")}
    n_alert = n_true_alert = 0
    caught, leads = [], []
    dog_days = 0
    work = pd.DataFrame(dict(dog_id=df.dog_id.values, day=df.day.values, s=score, ok=df[f"eval_{cond}"].values))
    for dog_id, g in work.groupby("dog_id", sort=False):
        g = g[g.ok]
        if g.empty:
            continue
        dog_days += len(g)
        alerts, last = [], -10_000
        for day, s in zip(g.day.values, g.s.values):
            if np.isfinite(s) and s >= thr and day - last >= refr:
                alerts.append(day); last = day
        wins = win_by_dog.get(dog_id, [])
        for a in alerts:
            n_alert += 1
            if any(s0 <= a < e0 for s0, e0 in wins):
                n_true_alert += 1
        dmin, dmax = g.day.min(), g.day.max()
        for s0, e0 in wins:
            if e0 <= dmin or s0 > dmax:
                continue                                   # window not observable for this dog
            hit = [a for a in alerts if s0 <= a < e0]
            caught.append(bool(hit))
            if hit:
                leads.append(e0 - min(hit))
    n_false = n_alert - n_true_alert
    return dict(
        events=len(caught),
        sensitivity=float(np.mean(caught)) if caught else np.nan,
        alert_precision=float(n_true_alert / n_alert) if n_alert else np.nan,
        false_alerts_per_dog_month=float(n_false / max(dog_days / 30, 1e-9)),
        median_lead_days=float(np.median(leads)) if leads else np.nan,
        alerts=n_alert,
    )


def calibrate_threshold(df_val, score_val, cond, events, budget=MAX_FALSE_PER_DOG_MONTH):
    """Choose the alert threshold on VALIDATION dogs: among thresholds whose false-alert rate stays within
    the budget, maximise the event-level F1 of (sensitivity, alert precision)."""
    s = score_val[np.isfinite(score_val)]
    if len(s) == 0:
        return np.inf
    cands = np.unique(np.quantile(s, np.linspace(0.80, 0.9995, 40)))
    best, best_f = cands[-1], -1.0
    for thr in cands:
        m = simulate_alerts(df_val, score_val, thr, cond, events)
        if m["false_alerts_per_dog_month"] > budget:
            continue
        se, pr = np.nan_to_num(m["sensitivity"]), np.nan_to_num(m["alert_precision"])
        f = 2 * se * pr / (se + pr) if se + pr > 0 else 0.0
        if f > best_f:
            best, best_f = thr, f
    return float(best)


def summarise(rows):
    return pd.DataFrame(rows)
