"""
Visual audit of the simulation + models. Writes PNGs to reports/audit/.

    python scripts/audit_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from praniiva.params import ACTIVITY_AGE_CURVE, CONDITIONS, positive_window  # noqa: E402

DATA, REP = ROOT / "data", ROOT / "reports"
OUT = REP / "audit"
OUT.mkdir(exist_ok=True)
KEY = {"oa_mobility": "high_int_min", "pruritic_flare": "scratch_min", "chf_decomp": "sleep_rr",
       "acute_gi": "eat_min", "tick_fever": "active_min", "ckd_pupd": "drink_min"}
plt.rcParams.update({"figure.dpi": 80, "font.size": 9})

co = pd.read_parquet(DATA / "cohort.parquet")
m = pd.read_parquet(DATA / "measured.parquet").sort_values(["dog_id", "day"])
tr = pd.read_parquet(DATA / "truth.parquet").sort_values(["dog_id", "day"])
ev = pd.read_parquet(DATA / "events.parquet")
ep = pd.read_parquet(DATA / "episodes.parquet")
R = pd.read_parquet(DATA / "risk_oof.parquet").sort_values(["dog_id", "day"])
D = int(m.day.max()) + 1


def series(df, dog, col):
    s = df[df.dog_id == dog].set_index("day")[col].reindex(range(D))
    return s


# ---------------------------------------------------------------- 1. event-aligned averages
fig, axes = plt.subplots(2, 6, figsize=(22, 7))
for j, (cond, ch) in enumerate(KEY.items()):
    chronic = CONDITIONS[cond].get("chronic")
    lo, hi = (-180, 20) if chronic else (-30, 10)
    lags = np.arange(lo, hi)
    Z, RK, AL = [], [], []
    for _, e in ev[ev.condition == cond].iterrows():
        s = series(m, e.dog_id, ch)
        base = s.iloc[max(0, e.onset_day - 60):max(1, e.onset_day - 10)]
        if base.notna().sum() < 10:
            continue
        mu, sd = base.median(), max(base.std(), 1e-3)
        idx = e.event_day + lags
        ok = (idx >= 0) & (idx < D)
        z = np.full(len(lags), np.nan); z[ok] = ((s.values[idx[ok]] - mu) / sd)
        Z.append(z)
        r = series(R, e.dog_id, f"risk_{cond}").values
        a = series(R, e.dog_id, f"fired_{cond}").fillna(False).astype(float).values
        rr = np.full(len(lags), np.nan); rr[ok] = r[idx[ok]]; RK.append(rr)
        aa = np.full(len(lags), np.nan); aa[ok] = a[idx[ok]]; AL.append(aa)
    Z, RK, AL = map(np.array, (Z, RK, AL))
    ax = axes[0, j]
    ax.plot(lags, pd.Series(np.nanmedian(Z, 0)).rolling(3, min_periods=1).mean(), c="k")
    ax.fill_between(lags, np.nanpercentile(Z, 25, 0), np.nanpercentile(Z, 75, 0), color="gray", alpha=.3)
    ax.axvline(0, c="r"); ax.axhline(0, c="b", lw=.5)
    ax.set_title(f"{cond}\n{ch} z vs pre-onset (n={len(Z)})")
    ax = axes[1, j]
    ax.plot(lags, np.nanmean(RK, 0), c="tab:red", label="mean risk")
    ax.plot(lags, np.nanmean(AL, 0) * 5, c="tab:blue", label="alerts/day x5")
    ax.axvline(0, c="r"); ax.set_ylim(0, 1.05); ax.set_xlabel("days relative to event")
    ax.legend(fontsize=7)
fig.suptitle("1. Event-aligned: does the signal (top) and the model risk (bottom) rise BEFORE the event (red line)?")
fig.tight_layout(); fig.savefig(OUT / "1_event_aligned.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. false alerts: what causes them
rows = []
win = {}
for _, e in ev[ev.condition.isin(CONDITIONS)].iterrows():
    a, b = positive_window(e.condition, e.event_day, e.onset_day)
    win.setdefault((e.dog_id, e.condition), []).append((a, b))
epd = {}
for _, e in ep.iterrows():
    epd.setdefault(e.dog_id, []).append((e.kind, e.start_day, e.start_day + e.length))
for cond in CONDITIONS:
    f = R[R[f"fired_{cond}"] == True]  # noqa: E712
    for dog, day in zip(f.dog_id, f.day):
        H = CONDITIONS[cond]["horizon"]
        true = any(a <= day < b for a, b in win.get((dog, cond), []))
        other = [c for c in CONDITIONS if c != cond and any(a - 7 <= day < b + 7 for a, b in win.get((dog, c), []))]
        epis = [k for k, s, t in epd.get(dog, []) if s - 3 <= day <= t + 3]
        cause = "TRUE" if true else (f"other illness: {other[0]}" if other else (f"look-alike: {epis[0]}" if epis else "unexplained"))
        rows.append(dict(cond=cond, cause=cause, dog=dog))
A = pd.DataFrame(rows)
tab = A.groupby(["cond", "cause"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 6))
tab.plot.barh(stacked=True, ax=ax, colormap="tab20")
ax.set_title("2. Why did each alert fire? (TRUE = within the condition's warning window)")
ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
fig.tight_layout(); fig.savefig(OUT / "2_alert_causes.png"); plt.close(fig)
tab.to_csv(OUT / "2_alert_causes.csv")

# ---------------------------------------------------------------- 3. calibration
fig, axes = plt.subplots(1, 6, figsize=(22, 4))
for j, cond in enumerate(CONDITIONS):
    from praniiva.features import add_labels  # noqa: F401
    r = R[["dog_id", "day", f"risk_{cond}"]].dropna()
    lab = np.zeros(len(r))
    for (dog, c), ws in win.items():
        if c != cond:
            continue
        mk = (r.dog_id.values == dog)
        for a, b in ws:
            lab[mk & (r.day.values >= a) & (r.day.values < b)] = 1
    q = pd.qcut(r[f"risk_{cond}"].rank(method="first"), 10, labels=False)
    g = pd.DataFrame(dict(p=r[f"risk_{cond}"].values, y=lab, q=q)).groupby("q").mean()
    axes[j].plot(g.p, g.y, "o-"); axes[j].plot([0, 1], [0, 1], "k--", lw=.5)
    axes[j].set_title(f"{cond}"); axes[j].set_xlabel("predicted risk"); axes[j].set_ylabel("observed")
fig.suptitle("3. Calibration: does 'risk 0.8' mean 80%?")
fig.tight_layout(); fig.savefig(OUT / "3_calibration.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. physiology sanity vs literature
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
mm = m.merge(co[["dog_id", "age0", "weight", "brachy", "mmvd", "coat", "group"]], on="dog_id")
mm["age"] = mm.age0 + mm.day / 365
b = mm.groupby(pd.cut(mm.age, [0, 1, 2, 4, 6, 8, 10, 12, 16])).active_min.median()
axes[0, 0].plot([i.mid for i in b.index], b.values, "o-", label="sim measured median")
axes[0, 0].plot(*ACTIVITY_AGE_CURVE, "r--", label="O'Rourke 2025")
axes[0, 0].set_title("active min vs age"); axes[0, 0].legend()
for k, grp in mm.groupby(np.where(mm.brachy, "brachy", np.where(mm.mmvd, "MMVD", "other"))):
    axes[0, 1].hist(grp.sleep_rr.dropna(), bins=60, alpha=.5, density=True, label=k)
axes[0, 1].axvline(14.2, c="k", ls="--"); axes[0, 1].axvline(30, c="r", ls="--")
axes[0, 1].set_title("sleeping RR (lit: night median 14.2, threshold 30)"); axes[0, 1].legend()
tm = tr.merge(m[["dog_id", "day", "scratch_min", "eat_min", "active_min", "sleep_rr"]], on=["dog_id", "day"], suffixes=("_t", ""))
for ax, (t, s) in zip([axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1]],
                      [("scratch_min_t", "scratch_min"), ("eat_min_t", "eat_min"),
                       ("walk_min", "active_min"), ("sleep_rr_t", "sleep_rr")]):
    smp = tm.sample(5000, random_state=0)
    x = smp[t] + (smp["trot_min"] if t == "walk_min" else 0)
    ax.scatter(x, smp[s], s=2, alpha=.3); lim = np.nanpercentile(np.r_[x, smp[s]], 99.5)
    ax.plot([0, lim], [0, lim], "r--"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"measured vs true: {s}"); ax.set_xlabel("truth"); ax.set_ylabel("measured")
axes[1, 2].hist(m.groupby("dog_id").wear_frac.apply(lambda s: (s > 0).mean()), bins=30)
axes[1, 2].set_title("share of days collar worn, per dog")
cnt = ev.groupby("dog_id").size().reindex(co.dog_id, fill_value=0)
axes[1, 3].hist(cnt, bins=range(0, 9)); axes[1, 3].set_title("events per dog per 400 days")
fig.suptitle("4. Does the simulated world look like the literature?")
fig.tight_layout(); fig.savefig(OUT / "4_sanity.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. example timelines
fig, axes = plt.subplots(6, 2, figsize=(22, 20))
for j, (cond, ch) in enumerate(KEY.items()):
    dogs = ev[ev.condition == cond].dog_id.unique()
    for k in range(2):
        dog = dogs[k * 3 % len(dogs)]
        ax = axes[j, k]
        s = series(m, dog, ch)
        base = s.shift(8).rolling(28, min_periods=10).median()
        mad = (s.shift(8) - base).abs().rolling(28, min_periods=10).median() * 1.4826
        ax.fill_between(range(D), base - 2 * mad, base + 2 * mad, color="g", alpha=.2)
        ax.plot(s.values, c="k", lw=.7)
        ax2 = ax.twinx()
        ax2.plot(series(R, dog, f"risk_{cond}").values, c="tab:red", lw=1.2)
        fired = series(R, dog, f"fired_{cond}").fillna(False).values.astype(bool)
        ax2.plot(np.where(fired)[0], np.ones(fired.sum()) * 1.02, "^", c="tab:red")
        ax2.set_ylim(0, 1.1)
        for _, e in ev[(ev.dog_id == dog) & (ev.condition == cond)].iterrows():
            a, b = positive_window(cond, e.event_day, e.onset_day)
            ax.axvspan(a, b, color="orange", alpha=.15); ax.axvline(e.event_day, c="r")
        for _, e in ep[ep.dog_id == dog].iterrows():
            ax.axvspan(e.start_day, e.start_day + e.length, color="gray", alpha=.2)
        ax.set_title(f"{cond} | {dog} ({co.set_index('dog_id').loc[dog, 'breed']}) | {ch}")
fig.suptitle("5. Example dogs: black=measured, green=personal normal, orange=warning window, red=event, triangles=alerts, red line=risk")
fig.tight_layout(); fig.savefig(OUT / "5_examples.png"); plt.close(fig)

# ---------------------------------------------------------------- 6. alert burden per dog
fa = A[A.cause != "TRUE"].groupby("dog").size().reindex(co.dog_id, fill_value=0)
x = co.set_index("dog_id").assign(false_alerts=fa)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
x.groupby("group").false_alerts.mean().sort_values().plot.barh(ax=axes[0], title="false alerts per dog (400 d) by breed group")
x.groupby("coat").false_alerts.mean().plot.bar(ax=axes[1], title="by coat")
axes[2].hist(fa, bins=range(0, 15)); axes[2].set_title("false alerts per dog over 400 days")
fig.tight_layout(); fig.savefig(OUT / "6_alert_burden.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
print(tab.to_string())
