"""
Real collar data -> daily measurements -> data quality -> personal baseline -> concern levels / alerts.

Folder layout (written by tools/collar_logger.py or copied from the collar's microSD card):
    data/raw/<dog>/dog.json                     dog profile (see template below; created if missing)
    data/raw/<dog>/*_raw.csv                    timestamp_ms,ax,ay,az
    data/raw/<dog>/*_qc.csv                     per-minute collar telemetry (optional but recommended)

    python tools/process_collar_data.py --dog ID0001

Outputs data/real/<dog>_daily.parquet (all days) and prints today's status per condition.
The first 35 days are a LEARNING PERIOD: the dog's personal baseline does not exist yet, so conditions report
'insufficient_data' - that is correct behaviour, not a bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from praniiva.features import build_features               # noqa: E402
from praniiva.ingest import IngestReport, daily_from_raw    # noqa: E402
from praniiva.params import BREEDS, CONDITIONS              # noqa: E402
from praniiva.pipeline import debounce_alerts, score        # noqa: E402
from praniiva.quality import assess, clean, maintenance_messages   # noqa: E402
from praniiva.reference import explain                      # noqa: E402

TEMPLATE = dict(dog_id="ID0001", breed="Indian Pariah", weight_kg=22.0, birth_date="2020-01-01",
                vet_diagnosed_mmvd=False, diagnoses=[], notes="breed must be one of params.BREEDS")


def profile_row(prof: dict, first_date) -> dict:
    b = prof["breed"]
    if b not in BREEDS:
        sys.exit(f"dog.json breed '{b}' not in {list(BREEDS)}")
    info = BREEDS[b]
    age0 = (pd.Timestamp(first_date) - pd.Timestamp(prof["birth_date"])).days / 365.25
    return dict(dog_id=prof["dog_id"], breed=b, group=info["group"], weight=float(prof["weight_kg"]), age0=age0,
                coat=info["coat"], double_coat=info["double"], cfr=info["cfr"], brachy=info["cfr"] < 0.2,
                mmvd=bool(prof.get("vet_diagnosed_mmvd", False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", required=True)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    d = Path(a.raw_dir) / a.dog
    pj = d / "dog.json"
    if not pj.exists():
        d.mkdir(parents=True, exist_ok=True)
        pj.write_text(json.dumps({**TEMPLATE, "dog_id": a.dog}, indent=2))
        sys.exit(f"Created {pj} - fill in breed, weight, birth date and vet diagnoses, then run again.")
    prof = json.loads(pj.read_text())
    raws = sorted(d.glob("*_raw.csv"))
    qcs = sorted(d.glob("*_qc.csv"))
    if not raws:
        sys.exit(f"no *_raw.csv files in {d}")
    rep = IngestReport()
    daily = daily_from_raw(raws, a.dog, ROOT / "models", rep, qc_paths=qcs)
    print(f"[ingest] {len(raws)} raw file(s), {rep.rows_in:,} rows in, {rep.rows_used:,} used, {len(daily)} day(s)")
    for x in rep.fixes + rep.warnings:
        print(f"   - {x}")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day"] = (pd.to_datetime(daily.date) - pd.to_datetime(daily.date.min())).dt.days
    for c in ("wx_tmax", "wx_rh", "wx_hi"):
        daily[c] = np.nan                     # weather not connected yet (phone/API) - features tolerate NaN
    if "rr_seg_iqr" not in daily:
        daily["rr_seg_iqr"] = np.nan
    co = pd.DataFrame([profile_row(prof, daily.date.min())])
    q = assess(daily)
    cl = clean(q)
    F = build_features(cl, co, None)
    bundle = joblib.load(ROOT / "models" / "risk_models.joblib")
    dx = pd.DataFrame([dict(dog_id=a.dog, condition=x["condition"],
                            event_day=(pd.Timestamp(x["date"]) - pd.to_datetime(daily.date.min())).days)
                       for x in prof.get("diagnoses", [])], columns=["dog_id", "condition", "event_day"])
    sc = debounce_alerts(score(F, bundle), diagnoses=dx if len(dx) else None)
    out = Path(ROOT / "data" / "real"); out.mkdir(parents=True, exist_ok=True)
    q.merge(sc, on=["dog_id", "day"]).to_parquet(out / f"{a.dog}_daily.parquet")
    last = sc.iloc[-1]
    print(f"\n[{a.dog}] latest day {daily.date.iloc[-1]}  (day {int(last.day) + 1} of monitoring; "
          f"personal baseline needs ~35 days)")
    print(f"   wear coverage {daily.wear_frac.iloc[-1]:.0%} | day status {daily.day_status.iloc[-1]}")
    for c in CONDITIONS:
        st, conc = last[f"status_{c}"], last[f"concern_{c}"]
        alert = " <<< ALERT" if last.get(f"fired_{c}", False) else ""
        why = "; ".join(explain(F.iloc[-1], c)) if alert else ""
        print(f"   {CONDITIONS[c]['label'][:45]:<46} {st:<18} concern {conc if np.isfinite(conc) else float('nan'):.2f}{alert} {why}")
    msgs = maintenance_messages(q)
    if len(msgs):
        print("\n   device messages:")
        for _, r in msgs.tail(5).iterrows():
            print(f"   - day {r.day}: {r.message}")
    print(f"\nsaved {out / (a.dog + '_daily.parquet')}")


if __name__ == "__main__":
    main()
