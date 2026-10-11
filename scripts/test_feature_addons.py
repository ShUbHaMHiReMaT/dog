"""
Do ARIMA / Chronos surprise features help the RISK model on top of the baseline features
(rolling z, trends, CUSUM)? Adopt only if AUPRC improves by >= 0.02 on held-out dogs.

    python scripts/test_feature_addons.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from praniiva.bakeoff_risk import DETECTOR_TASKS, make_hgb, evaluate_risk_model   # noqa: E402
from praniiva.daily import DEFAULT_CONFIG, measure, simulate_cohort                # noqa: E402
from praniiva.features import add_labels, build_features, feature_columns          # noqa: E402
from praniiva.forecasters import arima_surprise, chronos_surprise                   # noqa: E402

REPORTS, MODELS = ROOT / "reports", ROOT / "models"


def main(n_dogs=400, sub_dogs=140):
    cohort, truth, events, _, _ = simulate_cohort(n_dogs, seed=0, verbose=False)
    meas = measure(truth, cohort, DEFAULT_CONFIG, MODELS)
    F = add_labels(build_features(meas, cohort, events), events, cohort)
    rng = np.random.default_rng(0)
    mmvd = cohort.loc[cohort.mmvd, "dog_id"].values
    keep = np.r_[mmvd, rng.choice(np.setdiff1d(cohort.dog_id, mmvd), sub_dogs - len(mmvd), replace=False)]
    sub = meas[meas.dog_id.isin(keep)]
    chans = sorted({c for c, _ in DETECTOR_TASKS.values()})
    print("computing ARIMA + Chronos features ...", flush=True)
    ar = arima_surprise(sub, chans)
    ch = chronos_surprise(sub, chans)
    G = F[F.dog_id.isin(keep)].merge(ar, on=["dog_id", "day"], how="left").merge(ch, on=["dog_id", "day"], how="left")
    base = feature_columns(F)
    ar_cols = [c for c in G.columns if c.endswith("__arima_z3")]
    ch_cols = [c for c in G.columns if c.endswith("__chronos_z3")]
    rows = []
    for cond in DETECTOR_TASKS:
        res = {}
        for name, cols in (("baseline features", base), ("+ ARIMA", base + ar_cols), ("+ Chronos", base + ch_cols)):
            r = evaluate_risk_model(G, cols, cond, events, make_hgb, "clf")
            res[name] = r
            rows.append(dict(condition=cond, features=name, auroc=r["auroc"], auprc=r["auprc"],
                             sensitivity=r["sensitivity"], alert_precision=r["alert_precision"]))
            print(f"  {cond:<15} {name:<18} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f} "
                  f"sens {r['sensitivity']:.2f} prec {r['alert_precision']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "feature_addons.csv", index=False)
    piv = df.pivot(index="condition", columns="features", values="auprc")
    verdict = {}
    for add in ("+ ARIMA", "+ Chronos"):
        gain = piv[add] - piv["baseline features"]
        helped = list(gain[gain >= 0.02].index)
        verdict[add] = dict(mean_gain=float(gain.mean()), helped=helped,
                            adopt=bool(len(helped) >= 1 and gain.mean() > 0))
    (REPORTS / "feature_addons_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
