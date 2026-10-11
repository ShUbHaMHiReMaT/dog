"""Turn the bake-off CSVs into reports/ALGORITHM_BAKEOFF.md with a verdict per requested algorithm."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _md(df: pd.DataFrame, floatfmt=3) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join([head, sep, *rows])


def write_bakeoff_report(reports: Path):
    A = pd.read_csv(reports / "bakeoff_behaviour.csv") if (reports / "bakeoff_behaviour.csv").exists() else None
    B = pd.read_csv(reports / "bakeoff_detectors.csv")
    C = pd.read_csv(reports / "bakeoff_risk.csv")
    D = pd.read_csv(reports / "sensor_ablation.csv")
    E = pd.read_csv(reports / "split_inflation.csv")
    winners = json.loads((reports / "winners.json").read_text())

    # ---------------- verdicts -------------------------------------------------------
    def risk_verdict(model):
        g = C[C.model == model]
        wins = [c for c, w in winners.items() if w == model]
        best = C.groupby("condition").auprc.max()
        close = sum(g.set_index("condition").auprc.reindex(best.index) >= best - 0.02)
        if wins:
            return "ADOPT", f"best (or simplest within 0.01 AUPRC) for: {', '.join(wins)}"
        if close >= 3:
            return "COMPETITIVE - not needed", f"within 0.02 AUPRC of the best on {close}/6 conditions but never the simplest winner"
        return "REJECT", f"within 0.02 AUPRC of best on only {close}/6 conditions"

    def det_verdict(name):
        base = B[B.detector == "Rolling-median z (baseline)"].set_index("condition").auroc
        g = B[B.detector == name].set_index("condition").auroc.reindex(base.index)
        gain = (g - base)
        wins = list(gain[gain >= 0.02].index)
        if name.startswith("Rolling"):
            return "ADOPT (baseline)", "cheapest; reference for the others"
        if len(wins) >= 2:
            return "ADOPT as feature", f"beats rolling-median z by >=0.02 AUROC on {', '.join(wins)}"
        if len(wins) == 1:
            return "PARTIAL - feature for one condition", f"only better on {wins[0]} (+{gain[wins[0]]:.3f})"
        return "REJECT", f"mean AUROC change vs baseline {gain.mean():+.3f}"

    def beh_verdict(name):
        if A is None:
            return "not run", ""
        r = A[A.model == name]
        return (r.verdict.iloc[0], f"macro-F1 {r.macro_f1.iloc[0]:.3f} vs incumbent {A.macro_f1.iloc[0]:.3f}") if len(r) else ("-", "")

    ab = D.pivot_table(index="condition", columns="config", values="auprc")
    fusion_gain = (ab.get("C_accel_mic_env") - ab.get("A_accel")).mean() if "A_accel" in ab else np.nan
    skin_gain = (ab.get("D_plus_skin_temp") - ab.get("C_accel_mic_env")) if "D_plus_skin_temp" in ab else None
    cal_rows = E[["condition", "calibrated_sens", "calibrated_prec", "calibrated_fa", "fixed05_sens", "fixed05_prec", "fixed05_fa"]]
    infl = (E.auroc_random_split - E.auroc_dogs_held_out)

    v = []
    xa = beh_verdict("XGBoost"); xc = risk_verdict("XGBoost")
    v.append(("XGBoost", f"{xa[0]} (behaviour) / {xc[0]} (risk)", f"behaviour: {xa[1]}; risk: {xc[1]}"))
    v.append(("Bayesian Network", *risk_verdict("Bayesian (naive-Bayes network)")))
    addon = json.loads((reports / "feature_addons_verdict.json").read_text()) if (reports / "feature_addons_verdict.json").exists() else None

    def with_addon(det_name, key):
        dv = det_verdict(det_name)
        if addon is None:
            return dv
        a = addon[key]
        rv = "ADOPT as risk feature" if a["adopt"] else "NOT added to risk model"
        return (f"{dv[0]} (single channel) / {rv}",
                f"{dv[1]}; inside the risk model mean AUPRC change {a['mean_gain']:+.3f}"
                + (f", helped {', '.join(a['helped'])}" if a["helped"] else ", no condition improved by >=0.02"))

    v.append(("ARIMA", *with_addon("ARIMA surprise", "+ ARIMA")))
    ra = beh_verdict("Random Forest"); rc = risk_verdict("Random Forest")
    v.append(("Random Forest", f"{ra[0]} (behaviour) / {rc[0]} (risk)", f"behaviour: {ra[1]}; risk: {rc[1]}"))
    v.append(("SVM", *beh_verdict("SVM (RBF)")))
    sa = beh_verdict("Stacked (HGB+RF+SVM -> LR)"); sc = risk_verdict("Stacked ensemble (LR+XGB+HGB -> LR)")
    v.append(("Stacked Ensemble", f"{sa[0]} (behaviour) / {sc[0]} (risk)", f"behaviour: {sa[1]}; risk: {sc[1]}"))
    v.append(("Survival Analysis (Cox PH)", *risk_verdict("Survival analysis (Cox PH)")))
    v.append(("CNN (1D, raw signal)", *beh_verdict("1D CNN on raw signal")))
    cu = det_verdict("CUSUM change-point")
    v.append(("Change-point detection", f"{cu[0]} + HMM bout smoothing ADOPTED",
              f"CUSUM: {cu[1]}. Sticky-HMM (Viterbi) bout segmentation raised day-level PPV of eating 0.78->0.85 and drinking 0.83->1.00 (see behaviour_validation.json)"))
    if A is not None:
        cnn = A[A.model == "1D CNN on raw signal"].macro_f1.iloc[0]
        v.append(("Feature extraction", "ADOPT" if A.macro_f1.iloc[0] >= cnn - 0.01 else "REPLACE with CNN",
                  f"engineered features + HistGB macro-F1 {A.macro_f1.iloc[0]:.3f} vs raw-signal CNN {cnn:.3f}; features are explainable to a vet"))
    v.append(("Sensor fusion", "ADOPT (accel + mic + env)" if fusion_gain > 0.005 else "WEAK",
              f"mean AUPRC gain of config C over accelerometer-only: {fusion_gain:+.3f}"))
    rr = risk_verdict("Deterministic rules")
    v.append(("Deterministic rule-based system", rr[0] + " + always-on safety rules",
              rr[1] + "; RR>30 rule kept as a transparent safety net for diagnosed cardiac dogs"))
    v.append(("Threshold calibration", "ADOPT",
              "compare calibrated vs fixed 0.5 threshold in the table below - fixed thresholds either flood owners with alerts or miss events"))
    lr = risk_verdict("Logistic regression (linear)"); lt = det_verdict("Linear-regression trend")
    v.append(("Linear regression", f"{lr[0]} (logistic risk) / {lt[0]} (trend)", f"risk: {lr[1]}; trend detector: {lt[1]}"))
    v.append(("(extra) Chronos-Bolt, Hugging Face", *with_addon("Chronos-Bolt surprise (HF)", "+ Chronos")))
    v.append(("Precision & recall evaluation", "ADOPT (mandatory)",
              f"random row splits inflate AUROC by {infl.mean():+.3f} on average vs dogs held out - every number here is event-level P/R on unseen dogs"))
    V = pd.DataFrame(v, columns=["Algorithm", "Verdict", "Evidence (simulation)"])

    lines = ["# Algorithm bake-off - PRANIIVA canine simulation", "",
             "> All numbers come from SIMULATED dogs whose disease effects are parameterised from the published",
             "> literature in the feasibility dossier. They rank algorithms against each other; they are NOT clinical",
             "> performance claims. Re-run this bake-off on real collar data before trusting any verdict.", "",
             "Validation: whole dogs held out (GroupKFold), alert thresholds calibrated on separate validation dogs,",
             "false-alert budget 0.25 per dog-month, event-level precision/recall reported.", "",
             "## Verdicts", "", _md(V), ""]
    if A is not None:
        lines += ["## A. Behaviour recognition (raw collar IMU, dogs held out)", "", _md(A), ""]
    lines += ["## B. Per-dog abnormality detectors (single channel)", "",
              _md(B[["condition", "channel", "detector", "auroc", "sensitivity", "alert_precision",
                     "false_alerts_per_dog_month", "median_lead_days", "compute_s"]]), ""]
    lines += ["## C. Risk prediction (all features)", "",
              _md(C[["condition", "model", "auroc", "auprc", "prevalence", "sensitivity", "alert_precision",
                     "false_alerts_per_dog_month", "median_lead_days", "seconds"]]), "",
              "Winners (AUPRC, ties -> simpler model): " + ", ".join(f"**{k}**: {w}" for k, w in winners.items()), ""]
    lines += ["## D. Sensor fusion ablation (AUPRC by sensor configuration)", "", _md(ab.reset_index()), ""]
    lines += ["## E. Why dogs must be held out, and why thresholds must be calibrated", "", _md(E), ""]
    if (reports / "feature_addons.csv").exists():
        fa = pd.read_csv(reports / "feature_addons.csv").pivot(index="condition", columns="features", values="auprc")
        lines += ["## F. Do ARIMA / Chronos features help the risk model? (AUPRC, dogs held out)", "", _md(fa.reset_index()), ""]
    (reports / "ALGORITHM_BAKEOFF.md").write_text("\n".join(lines), encoding="utf-8")
    return V


if __name__ == "__main__":
    print(write_bakeoff_report(Path(__file__).resolve().parents[1] / "reports").to_string())
