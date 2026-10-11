"""
Build everything the dashboard needs:
  * (re)train the Tier 1 behaviour model + RR + bout-smoothing studies   (--retrain-behaviour)
  * simulate the cohort, measure with the chosen sensor config, build features
  * train production risk models using the bake-off winners (reports/winners.json)
  * save data/*.parquet and models/risk_models.joblib

    python scripts/run_simulation.py                  # uses existing behaviour model
    python scripts/run_simulation.py --retrain-behaviour --dogs 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from praniiva.behaviour import run_behaviour_study, run_rr_study, run_sequence_study   # noqa: E402
from praniiva.daily import DEFAULT_CONFIG, measure, simulate_cohort                     # noqa: E402
from praniiva.features import add_labels, build_features                                 # noqa: E402
from praniiva.params import CONDITIONS                                                   # noqa: E402
from praniiva.pipeline import debounce_alerts, train_production                          # noqa: E402
from praniiva.quality import assess, clean, detection_report, maintenance_messages       # noqa: E402
from praniiva.reference import build_reference, explain                                  # noqa: E402
import pandas as pd                                                                       # noqa: E402

MODELS, DATA, REPORTS = ROOT / "models", ROOT / "data", ROOT / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogs", type=int, default=400)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--retrain-behaviour", action="store_true")
    a = ap.parse_args()
    for p in (MODELS, DATA, REPORTS):
        p.mkdir(exist_ok=True)
    t0 = time.time()
    if a.retrain_behaviour or not (MODELS / "behaviour_model.joblib").exists():
        print("[1] Tier 1: behaviour model, RR algorithm, bout smoothing")
        run_behaviour_study(MODELS)
        run_rr_study(MODELS)
        run_sequence_study(MODELS)
    print(f"[2] Simulating {a.dogs} dogs")
    cohort, truth, events, episodes, wx = simulate_cohort(a.dogs, seed=0)
    meas_raw = measure(truth, cohort, a.config, MODELS, faults=True)      # real-world faults included
    meas_q = assess(meas_raw)                                              # v0.2 data-quality grading
    meas = clean(meas_q)
    maint = maintenance_messages(meas_q)
    print(f"    QC: {len(maint)} owner maintenance messages; fault detection:")
    print(detection_report(meas_q).round(2).to_string(index=False))
    F = add_labels(build_features(meas, cohort, events), events, cohort)
    wfile = REPORTS / "winners.json"
    winners = json.loads(wfile.read_text()) if wfile.exists() else {c: "HistGB" for c in CONDITIONS}
    print(f"[3] Training production models: {winners}")
    bundle, oof = train_production(F, events, winners, MODELS)
    oof = debounce_alerts(oof)
    print("[4] Saving data/")
    cohort.to_parquet(DATA / "cohort.parquet")
    truth.to_parquet(DATA / "truth.parquet")
    meas_q.to_parquet(DATA / "measured.parquet")            # raw values + dq flags (dashboard marks bad days)
    maint.to_parquet(DATA / "maintenance.parquet")
    build_reference(meas, cohort, MODELS / "reference.parquet")
    fired = []
    Fi = F.set_index(["dog_id", "day"])
    for cond in CONDITIONS:
        col = f"fired_{cond}"
        if col not in oof:
            continue
        for _, r in oof[oof[col] == True].iterrows():  # noqa: E712
            fired.append(dict(dog_id=r.dog_id, day=int(r.day), condition=cond, risk=float(r[f"risk_{cond}"]),
                              why="; ".join(explain(Fi.loc[(r.dog_id, r.day)], cond))))
    pd.DataFrame(fired).to_parquet(DATA / "alerts.parquet")
    events.to_parquet(DATA / "events.parquet")
    episodes.to_parquet(DATA / "episodes.parquet")
    oof.to_parquet(DATA / "risk_oof.parquet")
    keep = ["dog_id", "day"] + [c for c in F.columns if c.endswith(("__z7", "__m7", "__zlong"))]
    F[keep].to_parquet(DATA / "features_slim.parquet")
    (DATA / "run_info.json").write_text(json.dumps(dict(dogs=a.dogs, config=a.config, winners=winners), indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min.  Next:  streamlit run app.py")


if __name__ == "__main__":
    main()
