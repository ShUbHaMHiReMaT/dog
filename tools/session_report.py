"""
Analyse one live session recorded by the dashboard / LiveCollar and write a summary (JSON + PNG figures).

    python tools/session_report.py data/raw/MD_WRIST            # newest session in that folder
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from praniiva.live import ACTIVITIES  # noqa: E402

COLORS = {"standing": "#72b7b2", "sitting": "#9467bd", "lying_belly": "#4c78a8", "lying_side": "#1f3f6e",
          "tilted": "#8c8c8c", "still": "#9d9d9d", "small": "#d4a017", "walking": "#54a24b",
          "running": "#e45756", "large": "#f58518", "shaking": "#b279a2"}


def analyse(folder: Path, out: Path):
    base = sorted(folder.glob("*_session.json"))[-1].with_name(sorted(folder.glob("*_session.json"))[-1].name
                                                                 .replace("_session.json", ""))
    rd = lambda k: pd.read_csv(f"{base}_{k}.csv")
    raw, lab, snd, qc, ev = rd("raw"), rd("labels"), rd("sound"), rd("qc"), rd("events")
    meta = json.loads(Path(f"{base}_session.json").read_text())
    t0 = raw.timestamp_ms.iloc[0]
    dur = (raw.timestamp_ms.iloc[-1] - t0) / 1000
    d = np.diff(raw.timestamp_ms.values)
    lost = int(sum(int(x // 20) - 1 for x in d[d > 30]))
    A = raw[["ax", "ay", "az"]].to_numpy()
    vm = np.linalg.norm(A, axis=1)
    acts = lab.activity.value_counts()
    bouts = []
    for a in lab.activity:
        if bouts and bouts[-1][0] == a:
            bouts[-1][1] += 2
        else:
            bouts.append([a, 2])
    summary = dict(
        session=base.name, mount=meta.get("mount"), device=meta.get("device"),
        duration_min=round(dur / 60, 2), samples=len(raw), expected_samples=int(dur * 50) + 1,
        completeness_pct=round(100 * len(raw) / (int(dur * 50) + 1), 2), lost_samples=lost,
        longest_gap_ms=int(d.max()) if len(d) else 0,
        duplicate_timestamps=int((d == 0).sum()),
        read_errors=int(pd.to_numeric(qc.read_errors, errors="coerce").sum()) if len(qc) else None,
        sensor_recoveries=int(pd.to_numeric(qc.recoveries, errors="coerce").sum()) if len(qc) else None,
        dropped=int(pd.to_numeric(qc.dropped, errors="coerce").sum()) if len(qc) else None,
        qc_reports=len(qc), reconnects=meta.get("reconnects"),
        accel_peak_g=round(float(vm.max()), 2), accel_range_g=8,
        clipping_samples=int((np.abs(A) >= 7.9).sum()),
        activity_seconds={ACTIVITIES[k]: int(v * 2) for k, v in acts.items()},
        bouts_over_10s=[(ACTIVITIES[a], s) for a, s in bouts if s >= 10],
        n_activity_changes=len(bouts) - 1,
        sound=dict(rows=len(snd), mic_ok_pct=round(100 * snd.mic_ok.mean(), 1) if len(snd) else None,
                   median_dbfs=round(float(snd.rms_dbfs.median()), 1) if len(snd) else None,
                   loudest_peak_dbfs=round(float(snd.peak_dbfs.max()), 1) if len(snd) else None),
        markers=ev.label.tolist(),
        model_vs_rules_agreement=None,
    )
    m = {"lying_rest": ("still", "lying_belly", "lying_side"), "walking": ("walking",), "trotting": ("walking", "running"),
         "running": ("running",), "shaking": ("shaking",), "scratching": ("shaking", "small"), "standing": ("standing", "still"),
         "sitting": ("sitting", "still"), "eating": ("small",)}
    agree = [r.activity in m.get(r.model_label, ()) for r in lab.itertuples()]
    summary["model_vs_rules_agreement"] = round(100 * float(np.mean(agree)), 1) if agree else None

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1.1]))
    t = (raw.timestamp_ms - t0) / 60000
    ax[0].plot(t, vm, lw=0.4, color="#333")
    ax[0].set_ylabel("|acceleration| (g)")
    ax[0].set_title(f"{base.name} - {summary['duration_min']} min, {summary['completeness_pct']}% of samples received")
    lt = (lab.timestamp_ms - t0) / 60000
    keys = [k for k in ACTIVITIES if k in set(lab.activity)]
    for i, k in enumerate(keys):
        sel = lab.activity == k
        ax[1].scatter(lt[sel], [i] * int(sel.sum()), s=10, marker="s", color=COLORS[k])
    ax[1].set_yticks(range(len(keys)), [ACTIVITIES[k] for k in keys], fontsize=8)
    ax[1].set_title("Activity every 2 s (rules)", fontsize=10)
    if len(snd):
        st = (snd.timestamp_ms - t0) / 60000
        ax[2].plot(st, snd.peak_dbfs, lw=0.6, color="#e45756", label="peak")
        ax[2].plot(st, snd.rms_dbfs, lw=0.9, color="#4c78a8", label="average")
        ax[2].legend(fontsize=8, loc="upper right")
    ax[2].set_ylabel("sound (dBFS)")
    ax[2].set_xlabel("minutes from start")
    for e in ev.itertuples():
        for a_ in ax:
            a_.axvline((e.timestamp_ms - t0) / 60000, color="#999", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "session_overview.png", dpi=110)
    plt.close(fig)
    fig, a2 = plt.subplots(figsize=(6, 3.2))
    s = pd.Series(summary["activity_seconds"]).sort_values()
    a2.barh(s.index, s.values / 60, color=[COLORS[k] for k in ACTIVITIES if ACTIVITIES[k] in s.index][::-1]
            if False else "#4c78a8")
    a2.set_xlabel("minutes")
    a2.set_title("Time in each activity")
    fig.tight_layout()
    fig.savefig(out / "activity_budget.png", dpi=110)
    plt.close(fig)
    (out / "session_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/MD_WRIST")
    s = analyse(folder, Path("reports") / "sessions" / folder.name)
    print(json.dumps(s, indent=2, default=str))
