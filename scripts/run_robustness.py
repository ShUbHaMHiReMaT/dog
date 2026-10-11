"""
v0.2 robustness experiment: does the data-quality layer (and owner feedback) earn its place?

  S0  no sensor faults, no QC          (the optimistic world v0.1 was built in)
  S1  real-world faults, no QC         (v0.1 meeting reality)
  S2  real-world faults + QC layer     (v0.2)
  S2+feedback  owner dismisses alerts it cannot see a reason for -> per-dog threshold relaxes

    python scripts/run_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from praniiva.bakeoff_risk import evaluate_risk_model, make_hgb            # noqa: E402
from praniiva.daily import DEFAULT_CONFIG, measure, simulate_cohort         # noqa: E402
from praniiva.evaluation import REFRACTORY, calibrate_threshold              # noqa: E402
from praniiva.features import add_labels, build_features, feature_columns    # noqa: E402
from praniiva.params import CONDITIONS, positive_window                      # noqa: E402
from praniiva.quality import assess, clean, detection_report                 # noqa: E402

MODELS, REPORTS = ROOT / "models", ROOT / "reports"


def fault_alerts(Fe, scores, thr, meas, cond, events):
    """Count alerts that land on a faulty day and are not true warnings."""
    f = meas.set_index(["dog_id", "day"])["_fault"]
    wins = {}
    for _, e in events[events.condition == cond].iterrows():
        wins.setdefault(e.dog_id, []).append(positive_window(cond, e.event_day, e.onset_day))
    n = 0
    for dog, g in pd.DataFrame(dict(dog=Fe.dog_id, day=Fe.day, s=scores)).groupby("dog"):
        last = -999
        for day, s in zip(g.day, g.s):
            if np.isfinite(s) and s >= thr and day - last >= REFRACTORY.get(cond, 7):
                last = day
                if f.get((dog, day), "") and not any(a <= day < b for a, b in wins.get(dog, [])):
                    n += 1
    return n


def feedback_sim(Fe, scores, thr, cond, events, rng, p_dismiss_false=0.7, p_dismiss_true=0.3, relax=0.5, memory=60):
    """Owner sees each alert; dismisses false ones (and, realistically, some true early warnings it cannot see yet).
    A dismissal raises that dog's threshold toward 1 for `memory` days."""
    H = CONDITIONS[cond]["horizon"]
    wins = {}
    for _, e in events[events.condition == cond].iterrows():
        wins.setdefault(e.dog_id, []).append(positive_window(cond, e.event_day, e.onset_day))
    res = {}
    for policy in ("no feedback", "owner feedback"):
        n_alert = n_true = 0; caught = []; dog_days = 0
        for dog, g in pd.DataFrame(dict(dog=Fe.dog_id, day=Fe.day, s=scores)).groupby("dog"):
            dog_days += len(g)
            last, relaxed_until = -999, -1
            alerts = []
            for day, s in zip(g.day.values, g.s.values):
                t = thr + relax * (1 - thr) if (policy == "owner feedback" and day < relaxed_until) else thr
                if np.isfinite(s) and s >= t and day - last >= REFRACTORY.get(cond, 7):
                    last = day; alerts.append(day); n_alert += 1
                    true = any(a <= day < b for a, b in wins.get(dog, []))
                    n_true += true
                    if rng.random() < (p_dismiss_true if true else p_dismiss_false):
                        relaxed_until = day + memory
            for a, b in wins.get(dog, []):
                if b > g.day.min() and a <= g.day.max():
                    caught.append(any(a <= x < b for x in alerts))
        res[policy] = dict(sensitivity=float(np.mean(caught)) if caught else np.nan,
                           alert_precision=n_true / max(n_alert, 1),
                           false_alerts_per_dog_month=(n_alert - n_true) / max(dog_days / 30, 1))
    return res


def main(n_dogs=400):
    cohort, truth, events, _, _ = simulate_cohort(n_dogs, seed=0, verbose=False)
    scen = {}
    m0 = measure(truth, cohort, DEFAULT_CONFIG, MODELS, faults=False)
    scen["S0 no faults, no QC"] = (m0, m0)
    m1 = measure(truth, cohort, DEFAULT_CONFIG, MODELS, faults=True)
    scen["S1 faults, no QC (v0.1)"] = (m1, m1)
    a = assess(m1)
    detection_report(a).to_csv(REPORTS / "qc_detection.csv", index=False)
    scen["S2 faults + QC (v0.2)"] = (clean(a), m1)
    rows, fb_rows = [], []
    rng = np.random.default_rng(0)
    for name, (m, raw) in scen.items():
        F = add_labels(build_features(m, cohort, events), events, cohort)
        feats = feature_columns(F)
        for cond in CONDITIONS:
            r, Fe, sc = evaluate_risk_model(F, feats, cond, events, make_hgb, "clf", return_scores=True)
            ok = np.isfinite(sc)
            thr = calibrate_threshold(Fe[ok], sc[ok], cond, events)
            r["fault_false_alerts"] = fault_alerts(Fe[ok], sc[ok], thr, raw, cond, events)
            rows.append(dict(scenario=name, condition=cond, **r))
            print(f"  {name:<26} {cond:<15} AUROC {r['auroc']:.3f} sens {r['sensitivity']:.2f} "
                  f"prec {r['alert_precision']:.2f} FA/dog-mo {r['false_alerts_per_dog_month']:.3f} "
                  f"fault-caused FA {r['fault_false_alerts']}", flush=True)
            if name.startswith("S2"):
                fb = feedback_sim(Fe[ok], sc[ok], thr, cond, events, rng)
                for pol, v in fb.items():
                    fb_rows.append(dict(condition=cond, policy=pol, **v))
    df = pd.DataFrame(rows); df.to_csv(REPORTS / "robustness.csv", index=False)
    fb = pd.DataFrame(fb_rows); fb.to_csv(REPORTS / "owner_feedback.csv", index=False)
    print(df.pivot(index="condition", columns="scenario", values="false_alerts_per_dog_month").round(3).to_string())
    print(df.pivot(index="condition", columns="scenario", values="sensitivity").round(2).to_string())
    print(fb.round(3).to_string())


if __name__ == "__main__":
    main()
