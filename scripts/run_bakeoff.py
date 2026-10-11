"""
Run the full algorithm bake-off and write reports/ALGORITHM_BAKEOFF.md with ADOPT / REJECT verdicts.

    python scripts/run_bakeoff.py            # full (~30-60 min on a laptop CPU)
    python scripts/run_bakeoff.py --quick    # smaller cohort, for a smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from praniiva.bakeoff import bakeoff_behaviour                                  # noqa: E402
from praniiva.bakeoff_risk import (RISK_CANDIDATES, GroupStack, bakeoff_detectors, bakeoff_risk,  # noqa: E402
                                   evaluate_risk_model)
from praniiva.daily import DEFAULT_CONFIG, SENSOR_CONFIGS, measure, simulate_cohort  # noqa: E402
from praniiva.features import add_labels, build_features, feature_columns           # noqa: E402
from praniiva.params import CONDITIONS                                              # noqa: E402

MODELS, REPORTS, DATA = ROOT / "models", ROOT / "reports", ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-behaviour", action="store_true")
    args = ap.parse_args()
    for p in (MODELS, REPORTS, DATA):
        p.mkdir(exist_ok=True)
    n_dogs = 160 if args.quick else 400
    t00 = time.time()

    if not args.skip_behaviour:
        print("\n[A] Behaviour recognition bake-off (whole dogs held out)", flush=True)
        bakeoff_behaviour(REPORTS, n_dogs=30 if args.quick else 48, per_class=16 if args.quick else 24)

    print(f"\n[sim] cohort of {n_dogs} dogs", flush=True)
    cohort, truth, events, episodes, wx = simulate_cohort(n_dogs, seed=0)
    meas = measure(truth, cohort, DEFAULT_CONFIG, MODELS)
    F = add_labels(build_features(meas, cohort, events), events, cohort)
    feats = feature_columns(F)

    print("\n[B] Per-dog abnormality detectors", flush=True)
    bakeoff_detectors(F, meas, events, REPORTS, n_dogs_arima=60 if args.quick else 160)

    print("\n[C] Risk prediction bake-off (dogs held out, thresholds calibrated on separate dogs)", flush=True)
    risk = bakeoff_risk(F, feats, events, REPORTS)

    # ---- pick a winner per condition (AUPRC, tie -> simpler) ----------------------------
    simplicity = ["Deterministic rules", "Logistic regression (linear)", "Bayesian (naive-Bayes network)", "HistGB",
                  "XGBoost", "Random Forest", "Survival analysis (Cox PH)", "Stacked ensemble (LR+XGB+HGB -> LR)"]
    winners = {}
    for cond, g in risk.groupby("condition"):
        best = g.auprc.max()
        tied = g[g.auprc >= best - 0.01].copy()
        tied["rank"] = tied.model.map(simplicity.index)
        winners[cond] = tied.sort_values("rank").iloc[0].model
    print("    winners:", winners, flush=True)

    # ---- sensor fusion ablation with each condition's winner ---------------------------
    print("\n[D] Sensor-fusion ablation", flush=True)
    makers = {**RISK_CANDIDATES, "Stacked ensemble (LR+XGB+HGB -> LR)": GroupStack}
    rows = []
    for cfg in SENSOR_CONFIGS:
        m = measure(truth, cohort, cfg, MODELS)
        Fc = add_labels(build_features(m, cohort, events), events, cohort)
        fc = feature_columns(Fc)
        for cond in CONDITIONS:
            w = winners[cond]
            kind = "rule" if w == "Deterministic rules" else "stack" if w.startswith("Stacked") else \
                "clf" if w in makers else None
            if kind is None:      # Cox winner -> use HistGB for the ablation to keep runtime sane
                w, kind = "HistGB", "clf"
            r = evaluate_risk_model(Fc, fc, cond, events, makers.get(w), kind)
            rows.append(dict(config=cfg, label=SENSOR_CONFIGS[cfg]["label"], condition=cond, model=w, **r))
            print(f"    {cfg:<18} {cond:<15} AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f} sens {r['sensitivity']:.2f} "
                  f"prec {r['alert_precision']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(REPORTS / "sensor_ablation.csv", index=False)

    # ---- random split vs dog-held-out, for the winners (to show the inflation) ---------
    print("\n[E] Random row split vs held-out dogs", flush=True)
    rows = []
    for cond in CONDITIONS:
        w = winners[cond]
        if w in ("Deterministic rules", "Survival analysis (Cox PH)") or w.startswith("Stacked"):
            w = "HistGB"
        r_dog = evaluate_risk_model(F, feats, cond, events, makers[w], "clf")
        r_rnd = evaluate_risk_model(F, feats, cond, events, makers[w], "clf", split="random")
        r_fix = evaluate_risk_model(F, feats, cond, events, makers[w], "clf", fixed_threshold=0.5)
        rows.append(dict(condition=cond, model=w, auroc_dogs_held_out=r_dog["auroc"], auroc_random_split=r_rnd["auroc"],
                         auprc_dogs_held_out=r_dog["auprc"], auprc_random_split=r_rnd["auprc"],
                         calibrated_sens=r_dog["sensitivity"], calibrated_prec=r_dog["alert_precision"],
                         calibrated_fa=r_dog["false_alerts_per_dog_month"],
                         fixed05_sens=r_fix["sensitivity"], fixed05_prec=r_fix["alert_precision"],
                         fixed05_fa=r_fix["false_alerts_per_dog_month"]))
    pd.DataFrame(rows).to_csv(REPORTS / "split_inflation.csv", index=False)

    import json
    (REPORTS / "winners.json").write_text(json.dumps(winners, indent=2))
    from scripts_report import write_bakeoff_report   # noqa: E402
    write_bakeoff_report(REPORTS)
    print(f"\nDone in {(time.time() - t00) / 60:.1f} min -> {REPORTS / 'ALGORITHM_BAKEOFF.md'}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
