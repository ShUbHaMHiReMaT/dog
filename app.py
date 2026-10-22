"""
PRANIIVA canine health - simulation dashboard (clinician view first).

    streamlit run app.py

Everything shown is SIMULATED. Risk scores on the cohort pages are out-of-fold: the model that scored a
dog never saw that dog during training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from praniiva.daily import SENSOR_CONFIGS
from praniiva.params import BREEDS, CONDITIONS, RR_CHF_THRESHOLD, SIM_DAYS

ROOT = Path(__file__).parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"
st.set_page_config(page_title="PRANIIVA Canine Health Sim", page_icon="🐕", layout="wide")

CH_LABEL = {
    "active_min": "Active minutes", "high_int_min": "High-intensity minutes", "standing_min": "Standing minutes",
    "scratch_min": "Scratching minutes", "headshake_count": "Head shakes", "eat_min": "Eating minutes",
    "drink_min": "Drinking minutes", "pant_min": "Panting minutes", "night_bouts": "Night restlessness bouts",
    "gait_asym": "Gait asymmetry index", "sleep_rr": "Sleeping resp. rate (br/min)", "cough_count": "Cough events",
    "env_temp": "Collar ambient temp (C)", "skin_temp": "Skin temp (C, unvalidated)", "stiffness_latency": "Morning stiffness (min)",
}
COND_CH = {"oa_mobility": ["active_min", "high_int_min", "standing_min", "gait_asym"],
           "pruritic_flare": ["scratch_min", "headshake_count", "night_bouts"],
           "chf_decomp": ["sleep_rr", "cough_count", "active_min"],
           "acute_gi": ["eat_min", "active_min", "drink_min"],
           "tick_fever": ["active_min", "eat_min", "sleep_rr"],
           "ckd_pupd": ["drink_min", "eat_min", "night_bouts"]}
SHORT = {c: v["label"].split(" (")[0].split(" -")[0] for c, v in CONDITIONS.items()}


@st.cache_data
def load():
    if not (DATA / "risk_oof.parquet").exists():
        return None
    d = {k: pd.read_parquet(DATA / f"{k}.parquet") for k in
         ("cohort", "measured", "events", "episodes", "risk_oof", "truth")}
    for k in ("maintenance", "alerts"):
        f = DATA / f"{k}.parquet"
        d[k] = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    f = MODELS / "reference.parquet"
    d["reference"] = pd.read_parquet(f) if f.exists() else None
    d["info"] = json.loads((DATA / "run_info.json").read_text())
    return d


@st.cache_resource
def bundle():
    import joblib
    return joblib.load(MODELS / "risk_models.joblib")


def banner():
    st.caption("⚠️ **Research simulation - a monitoring & anomaly-alert prototype, not a diagnostic device.** "
               "No real-world detection performance has been established. "
               "Dogs, diseases and sensor errors are generated from published parameters "
               "(PRANIIVA feasibility dossier). Numbers rank methods and test the pipeline - they are not clinical claims.")


def personal_band(s: pd.Series):
    """Display band = the dog's normal from 2-4 months earlier (lagged 60 d, 60-day window), so a slow
    decline shows up as leaving the band instead of the band quietly following it down."""
    base = s.shift(60).rolling(60, min_periods=20).median()
    mad = (s.shift(60) - base).abs().rolling(60, min_periods=20).median() * 1.4826
    return base, base - 2 * mad, base + 2 * mad


def timeline_figure(meas, scores, events, episodes, channels, conds, today=None, chronos=None, truth_mode=True):
    rows = len(channels) + 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[CH_LABEL.get(c, c) for c in channels] + ["Concern level (1.0 = alert line - not a probability)"])
    x = meas.day.values
    for i, c in enumerate(channels, start=1):
        s = meas[c]
        base, lo, hi = personal_band(s)
        fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), i, 1)
        fig.add_trace(go.Scatter(x=x, y=lo, fill="tonexty", fillcolor="rgba(46,139,87,0.15)", line=dict(width=0),
                                 name="personal normal (±2 MAD)", showlegend=i == 1, hoverinfo="skip"), i, 1)
        fig.add_trace(go.Scatter(x=x, y=s, mode="lines", line=dict(width=1.3, color="#334"), name=CH_LABEL.get(c, c),
                                 showlegend=False), i, 1)
        if "dq_score" in meas:
            # mark only the days on which THIS channel was actually masked by the quality layer
            from praniiva.quality import clean
            masked = clean(meas)[c].isna().values & meas[c].notna().values
            bad = masked
            if bad.any():
                fig.add_trace(go.Scatter(x=x[bad], y=s.values[bad], mode="markers",
                                         marker=dict(symbol="x", size=6, color="darkorange"),
                                         name="untrustworthy reading (ignored by models)", showlegend=i == 1), i, 1)
        if c == "sleep_rr":
            fig.add_hline(y=RR_CHF_THRESHOLD, line=dict(dash="dot", color="crimson"), row=i, col=1)
        if chronos is not None and c in chronos:
            q10, q50, q90, d0 = chronos[c]
            xx = np.arange(d0, d0 + len(q50))
            fig.add_trace(go.Scatter(x=xx, y=q90, line=dict(width=0), showlegend=False, hoverinfo="skip"), i, 1)
            fig.add_trace(go.Scatter(x=xx, y=q10, fill="tonexty", fillcolor="rgba(70,110,220,0.2)", line=dict(width=0),
                                     name="Chronos 14-day forecast (10-90%)", showlegend=i == 1), i, 1)
            fig.add_trace(go.Scatter(x=xx, y=q50, line=dict(color="royalblue", dash="dash"), showlegend=False), i, 1)
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for k, cond in enumerate(conds):
        col = f"concern_{cond}"
        if col not in scores or scores[col].isna().all():
            continue
        yv = scores[col].clip(upper=2.0)
        fig.add_trace(go.Scatter(x=scores.day, y=yv, name=SHORT[cond], line=dict(color=colors[k % 6])), rows, 1)
        fired = scores[scores.get(f"fired_{cond}", False) == True]  # noqa: E712
        fig.add_trace(go.Scatter(x=fired.day, y=fired[col].clip(upper=2.0) + 0.06, mode="markers", marker=dict(symbol="triangle-up", size=11,
                                 color=colors[k % 6], line=dict(color="black", width=1)), name=f"ALERT {SHORT[cond]}",
                                 showlegend=False), rows, 1)
    if truth_mode and events is not None and len(events):
        for _, e in events.iterrows():
            for r in range(1, rows + 1):
                fig.add_vline(x=e.event_day, line=dict(color="crimson", width=1.5), row=r, col=1)
            fig.add_annotation(x=e.event_day, y=1.02, yref="paper", text=f"⚑ {SHORT.get(e.condition, e.condition)}",
                               showarrow=False, font=dict(color="crimson", size=11))
    if truth_mode and episodes is not None and len(episodes):
        for _, ep in episodes.iterrows():
            fig.add_vrect(x0=ep.start_day, x1=ep.start_day + ep.length, fillcolor="gray", opacity=0.10, line_width=0)
    if today is not None:
        for r in range(1, rows + 1):
            fig.add_vline(x=today, line=dict(color="black", dash="dash"), row=r, col=1)
    fig.update_layout(height=200 * rows + 80, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.03), hovermode="x unified")
    fig.add_hline(y=1.0, line=dict(color="white", dash="dot", width=1), row=rows, col=1)
    fig.update_yaxes(range=[0, 2.15], row=rows, col=1)
    return fig


# ======================================================================================
page = st.sidebar.radio("View", ["📡 Live collar", "🏥 Clinic overview", "🐕 Dog timeline", "🧪 Live simulator",
                                 "📊 Algorithms & validation", "🔧 Device design", "📚 Parameters"])
D = load()
st.sidebar.markdown("---")
st.sidebar.caption("PRANIIVA Technologies - canine health simulation v0.1")

if D is None and page not in ("🔧 Device design", "📚 Parameters", "📡 Live collar"):
    st.error("No simulation data yet. Run `python scripts/run_simulation.py` first.")
    st.stop()

# ======================================================================================

# ======================================================================================
BEH_NICE = {"lying_rest": "Lying / resting", "standing": "Standing", "walking": "Walking", "trotting": "Running / play",
            "scratching": "Scratching", "head_shaking": "Head shaking", "eating": "Eating", "drinking": "Drinking",
            "panting": "Panting"}
BEH_COLOR = {"lying_rest": "#4c78a8", "standing": "#9ecae9", "walking": "#54a24b", "trotting": "#2e7d32",
             "scratching": "#e45756", "head_shaking": "#f58518", "eating": "#b279a2", "drinking": "#72b7b2",
             "panting": "#eeca3b"}


@st.cache_resource
def live_sessions():
    return {}


if page == "📡 Live collar":
    from praniiva.live import ACTIVITIES, RULES
    ACT_COLOR = {"standing": "#72b7b2", "sitting": "#9467bd", "lying_belly": "#4c78a8", "lying_side": "#1f3f6e",
                 "tilted": "#8c8c8c", "still": "#9d9d9d", "small": "#d4a017", "walking": "#54a24b",
                 "running": "#e45756", "large": "#f58518", "shaking": "#b279a2"}
    ACT_ICON = {"standing": "🧍", "sitting": "🪑", "lying_belly": "🛌", "lying_side": "😴", "tilted": "↘",
                "still": "⏸", "small": "👃", "walking": "🚶", "running": "🏃", "large": "🤸", "shaking": "🌀"}
    st.title("📡 Live collar - real-time analysis")
    st.caption("Streaming from the PRANIIVA prototype (ESP32-S3 + BMI270, 50 samples/s, + sound level). "
               "Activity comes from transparent motion rules that work on a chest harness or a neck collar "
               "(thresholds provisional until tuned on labelled real recordings). Not a diagnostic device.")
    sess = live_sessions()
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    port = c1.text_input("Device port", "COM15")
    dog = c2.text_input("Dog ID", "ID0001")
    mount = c3.selectbox("Mounted at", ["chest", "neck collar", "back"], index=0)
    if c4.button("▶ Start live session", type="primary"):
        if sess.get("live"):
            sess["live"].stop()
        from praniiva.live import LiveCollar
        sess["live"] = LiveCollar(port, dog.strip() or "ID0001", mount=mount)
    if c5.button("■ Stop & save") and sess.get("live"):
        sess["live"].stop()
        st.success(f"Saved: {sess['live'].base}_*.csv")
        sess.pop("live")
    live = sess.get("live")
    if live is None:
        st.info("Connect the device by USB and press **Start live session**. Everything is recorded automatically "
                "to data/raw/<dog>/. (Close check_mic.py / check_sensor.py / collar_logger.py first - only one program "
                "can use the port.)")
        from praniiva.viz3d import dog3d_html
        st.markdown("**3D dog - preview of the postures the system can show** (follows the real device once a session runs)")
        st.iframe(dog3d_html(height=420, mount=mount, demo="cycle"), height=430)
        st.stop()
    b1, b1b, b2, b3 = st.columns([1.3, 1.3, 2, 0.8])
    if b1.button("① 🧍 Dog is STANDING still → calibrate",
                 help="Press after the dog has stood calmly for 3 s. Needed for any posture."):
        st.toast("Standing calibrated ✓ - now ask the dog to SIT and press ②" if live.calibrate_standing() else
                 "Not calibrated - the device was moving or no data yet; try again while the dog stands still")
    if b1b.button("② 🪑 Dog is SITTING still → calibrate",
                  help="Press after the dog has sat calmly for 3 s. Lets the system tell sitting from lying."):
        why = live.calibrate_sitting()
        st.toast("Sitting calibrated ✓ - posture detection is fully set up" if why is None else f"Not calibrated: {why}")
    label = b2.text_input("Mark what the dog is doing (saved with the recording)", "",
                          placeholder="e.g. walking on leash, eating, lying down, scratching")
    if b3.button("Add marker") and label:
        live.mark(label)
        st.toast(f"Marker saved: {label}")

    from praniiva.viz3d import dog3d_html
    st.iframe(dog3d_html(height=430, mount=live.mount), height=440)   # polls the live pose by itself

    @st.fragment(run_every="1s")
    def live_view():
        snap = live.snapshot(30)
        labels = snap["labels"]
        fmt = lambda v: "-" if v is None else f"{v:.0f}"
        fresh = snap["last_write"] is not None and time.time() - snap["last_write"] < 5
        if not snap["connected"]:
            st.error(f"🔌 Device disconnected - reconnecting every 2 s. Recording resumes automatically. "
                     f"({snap['error']})")
        elif snap["rate"] < 40:
            st.warning("Connected, waiting for motion data…")
        recent = [a for _, a, _, _ in labels[-2:]]
        now_act = recent[-1] if recent else None
        info = labels[-1][3] if labels else {}

        a1, a2 = st.columns([2, 3])
        with a1:
            if now_act:
                st.markdown(
                    f"<div style='padding:18px;border-radius:12px;background:{ACT_COLOR[now_act]};color:white'>"
                    f"<div style='font-size:15px;opacity:.85'>Doing now</div>"
                    f"<div style='font-size:34px;font-weight:700'>{ACT_ICON[now_act]} {ACTIVITIES[now_act]}</div>"
                    f"<div style='font-size:13px;opacity:.85'>motion {info.get('motion_g', 0):.3f} g · "
                    f"{info.get('dom_hz', 0)} Hz · tilt {fmt(info.get('tilt_deg'))}° · "
                    f"forward/back {fmt(info.get('pitch_deg'))}° · side {fmt(info.get('roll_deg'))}° · "
                    f"model says: {BEH_NICE.get(labels[-1][2], labels[-1][2])} (experimental)</div></div>",
                    unsafe_allow_html=True)
            else:
                st.info("First activity label after 2 s of data…")
            if snap["calibrated"] == "none":
                st.caption("Posture not set up: press **①** while the dog stands, then **②** while it sits.")
            elif snap["calibrated"] == "standing":
                st.caption("Standing calibrated. Press **②** while the dog sits to tell sitting from lying.")
            else:
                st.caption("Posture calibrated (standing + sitting) ✓")
        with a2:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Device", "🟢 live" if snap["connected"] and snap["rate"] > 40 else "🔴 offline")
            k2.metric("Samples / s", f"{snap['rate']:.1f}")
            k3.metric("Session", f"{int(snap['elapsed'] // 60)} min {int(snap['elapsed'] % 60)} s")
            k4.metric("Missing samples", f"{snap['gaps']}", f"{snap['reconnects']} reconnects", delta_color="off")
            rows = snap["rows"]
            st.markdown(f"**Recording** {'🟢 saving' if fresh else '🔴 not saving'} · "
                        f"{rows.get('raw', 0):,} motion rows · {rows.get('sound', 0):,} sound · "
                        f"{rows.get('labels', 0):,} labels · {rows.get('events', 0)} markers · "
                        f"{rows.get('qc', 0)} health reports")
            st.caption(f"Files: {snap['files']}")

        s = snap["samples"]
        if len(s) > 10:
            t = (s[:, 0] - s[-1, 0]) / 1000
            vm = np.linalg.norm(s[:, 1:4], axis=1)
            fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                                subplot_titles=("Motion, last 30 s (g)", "Time spent this session (min)"))
            for i, (nm, col) in enumerate((("x", "#e45756"), ("y", "#54a24b"), ("z", "#4c78a8"))):
                fig.add_trace(go.Scatter(x=t, y=s[:, 1 + i], name=nm, line=dict(width=1, color=col)), 1, 1)
            fig.add_trace(go.Scatter(x=t, y=vm, name="|a| total", line=dict(width=2, color="black")), 1, 1)
            bud = sorted(snap["budget"].items(), key=lambda kv: -kv[1])
            fig.add_trace(go.Bar(x=[v / 60 for _, v in bud], y=[ACTIVITIES[b] for b, _ in bud], orientation="h",
                                 marker_color=[ACT_COLOR[b] for b, _ in bud], showlegend=False), 1, 2)
            fig.update_xaxes(title_text="seconds ago", row=1, col=1)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch", key="live_main")

        if labels:
            lt = np.array([x[0] for x in labels], float)
            lt = (lt - lt[-1]) / 60000
            fig2 = go.Figure()
            for b in ACTIVITIES:
                m = np.array([x[1] == b for x in labels])
                if m.any():
                    fig2.add_trace(go.Scatter(x=lt[m], y=[ACTIVITIES[b]] * int(m.sum()), mode="markers",
                                              name=ACTIVITIES[b], marker=dict(symbol="square", size=9,
                                                                              color=ACT_COLOR[b])))
            fig2.update_layout(title="Activity timeline (each square = 2 s)", height=300, showlegend=False,
                               xaxis_title="minutes ago", margin=dict(l=10, r=10, t=40, b=10),
                               yaxis=dict(categoryorder="array", categoryarray=list(ACTIVITIES.values())))
            st.plotly_chart(fig2, width="stretch", key="live_timeline")

        if snap.get("sound"):
            snd = np.array(snap["sound"], float)
            st_t = snd[:, 0] - snd[-1, 0]
            figs = go.Figure([go.Scatter(x=st_t, y=snd[:, 2], name="peak", line=dict(color="#e45756", width=1)),
                              go.Scatter(x=st_t, y=snd[:, 1], name="average", line=dict(color="#4c78a8", width=2))])
            figs.update_layout(title=f"Sound level (now {snd[-1, 1]:.0f} dBFS) - levels only, no audio is recorded",
                               height=220, xaxis_title="seconds ago", yaxis_title="dBFS", yaxis=dict(range=[-90, 0]),
                               margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(figs, width="stretch", key="live_sound")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Resting breathing rate (experimental)**")
            if snap["rr"]:
                st.metric("Breaths / minute", f"{snap['rr'][-1][1]:.0f}", help="From body micro-motion during 60 s "
                          "of lying still. Not yet validated against counted breaths.")
            else:
                still = 0
                for _, a, _, _ in reversed(labels):
                    if a not in ("lying_belly", "lying_side", "still"):
                        break
                    still += 2
                st.info(f"Needs 60 s of continuous lying still (now: {min(still, 60)} s).")
        with r2:
            st.markdown("**Fit check** (is the device placed correctly?)")
            fit = snap.get("fit") or {}
            for k, nm in (("worn", "On the body"), ("loose", "Strap"), ("shifted", "Position since calibration"),
                          ("gravity", "Sensor reading")):
                v = str(fit.get(k, "checking"))
                icon = "🔴" if v.split(" ")[0] in ("NO", "FAULT", "LOOSE", "MOVED") else "⏳" if v.startswith(
                    ("checking", "calibrate")) else "🟢"
                st.write(f"{icon} {nm}: {v}")
            st.markdown("**Device health**")
            if snap["qc"]:
                q = snap["qc"][-1]
                try:
                    st.write(f"Resting jitter {float(q[2]):.2f} mg (loose fit -> high) · dropped {q[5]} · "
                             f"read errors {q[6]} · recoveries {q[8] if len(q) > 8 else '-'} (last minute)")
                except (ValueError, IndexError):
                    st.write(", ".join(q))
            else:
                st.write("First health report arrives after 1 minute.")
            for m_ in snap["messages"][-4:]:
                st.caption(m_)
        st.caption(f"Mounted at: {live.mount} · {snap['n']:,} samples received · rules: still < "
                   f"{RULES['still_std_g']} g, walking ≥ {RULES['small_std_g']} g, running > {RULES['run_std_g']} g, "
                   f"shaking ≥ {RULES['shake_min_hz']} Hz · on the chest, lying on the belly looks like standing - "
                   f"it is called 'lying on belly (likely)' after {RULES['belly_after_s'] // 60} min motionless")

    live_view()
    st.stop()

if page == "🏥 Clinic overview":
    st.title("Clinic overview")
    banner()
    today = st.slider("Simulation day ('today')", 40, SIM_DAYS - 1, 220)
    R = D["risk_oof"]
    co = D["cohort"].set_index("dog_id")
    snap = R[R.day == today].set_index("dog_id")
    rc = [f"concern_{c}" for c in CONDITIONS if f"concern_{c}" in snap]
    window = R[(R.day > today - 7) & (R.day <= today)]
    fired = [f"fired_{c}" for c in CONDITIONS if f"fired_{c}" in window]
    alerts = window.melt(id_vars=["dog_id", "day"], value_vars=fired, var_name="cond", value_name="f")
    alerts = alerts[alerts.f == True].copy()  # noqa: E712
    alerts["condition"] = alerts.cond.str.replace("fired_", "").map(SHORT)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dogs monitored", len(co))
    c2.metric("Alerts in last 7 days", len(alerts))
    c3.metric("Dogs wearing collar today", int(D["measured"].query("day == @today").wear_frac.gt(0).sum()))
    c4.metric("Diagnosed cardiac (RR-monitored)", int(co.mmvd.sum()))
    st.subheader("Alerts in the last 7 days")
    if len(alerts):
        a = alerts.merge(co[["breed", "age0", "weight"]], left_on="dog_id", right_index=True)
        a["age"] = (a.age0 + a.day / 365).round(1)
        if len(D["alerts"]):
            w = D["alerts"].assign(condition=D["alerts"].condition.map(SHORT))[["dog_id", "day", "condition", "concern", "why"]]
            a = a.merge(w, on=["dog_id", "day", "condition"], how="left")
        cols = ["dog_id", "breed", "age", "weight", "condition", "day"] + [c for c in ("concern", "why") if c in a]
        st.dataframe(a[cols].sort_values("day", ascending=False), hide_index=True, width="stretch")
    else:
        st.info("No alerts fired in this window.")
    mt = D["maintenance"]
    if len(mt):
        recent = mt[(mt.day > today - 7) & (mt.day <= today)]
        st.subheader(f"Device / data-quality messages sent to owners (last 7 days): {len(recent)}")
        st.caption("Bad data becomes a maintenance message - never a health alert.")
        st.dataframe(recent.sort_values("day", ascending=False), hide_index=True, width="stretch")
    st.subheader("Highest concern today")
    tab = snap[rc].copy()
    tab.columns = [SHORT[c.replace("concern_", "")] for c in rc]
    tab = tab.join(co[["breed", "weight"]])
    tab["max concern"] = tab[[SHORT[c.replace("concern_", "")] for c in rc]].max(axis=1)
    st.caption("Concern level: 1.0 = that condition's alert line. Empty = insufficient data or not applicable.")
    st.dataframe(tab.sort_values("max concern", ascending=False).head(25).style.format(precision=2)
                 .background_gradient(cmap="Reds", vmin=0, vmax=1.5,
                                      subset=[SHORT[c.replace('concern_', '')] for c in rc]), width="stretch")
    with st.expander("Ground truth (simulation only) - events near today"):
        ev = D["events"]
        st.dataframe(ev[(ev.event_day >= today - 7) & (ev.event_day <= today + 30)].sort_values("event_day"),
                     hide_index=True, width="stretch")

# ======================================================================================
elif page == "🐕 Dog timeline":
    st.title("Dog timeline")
    banner()
    co = D["cohort"]
    ev = D["events"]
    flag = st.checkbox("Only dogs with a simulated event", True)
    ids = sorted(ev.dog_id.unique()) if flag else sorted(co.dog_id)
    left, right = st.columns([1, 3])
    with left:
        dog_id = st.selectbox("Dog", ids)
        d = co.set_index("dog_id").loc[dog_id]
        st.markdown(f"**{d.breed}** · {d.weight} kg · {d.age0:.1f} y at start · coat {d.coat} ({d.coat_colour})")
        st.markdown(f"Diagnosed MMVD: **{'yes' if d.mmvd else 'no'}** · atopic: {'yes' if d.atopic else 'no'} · "
                    f"brachycephalic: {'yes' if d.brachy else 'no'}")
        dog_conds = [c for c in ev[ev.dog_id == dog_id].condition.unique() if c in CONDITIONS]
        conds = st.multiselect("Conditions", list(CONDITIONS), default=dog_conds or ["oa_mobility"],
                               format_func=lambda c: SHORT[c])
        default_ch = sorted({ch for c in conds for ch in COND_CH[c]}, key=list(CH_LABEL).index)
        chans = st.multiselect("Channels", list(CH_LABEL), default=default_ch[:5], format_func=CH_LABEL.get)
        truth_mode = st.checkbox("Show simulation ground truth (events ⚑, look-alike episodes ▒)", True)
        today = st.slider("Today (for Chronos forecast)", 40, SIM_DAYS - 15, 200)
        use_chronos = st.checkbox("Chronos-Bolt forecast from 'today' (Hugging Face)", False)
        st.dataframe(ev[ev.dog_id == dog_id], hide_index=True)
        eps = D["episodes"][D["episodes"].dog_id == dog_id]
        if len(eps):
            st.caption("Look-alike (non-disease) episodes")
            st.dataframe(eps, hide_index=True)
    with right:
        m = D["measured"][D["measured"].dog_id == dog_id].sort_values("day")
        sc = D["risk_oof"][D["risk_oof"].dog_id == dog_id].sort_values("day")
        chron = None
        if use_chronos and chans:
            try:
                from praniiva.forecasters import chronos_forecast_one
            except ImportError:
                st.warning("Chronos needs the research extras: pip install -r requirements-research.txt")
                chans_chronos = []
            else:
                chans_chronos = chans
            chron = {}
            for c in chans_chronos:
                hist = m[m.day < today][c].values
                if np.isfinite(hist).sum() > 20:
                    q10, q50, q90 = chronos_forecast_one(hist, 14)
                    chron[c] = (q10, q50, q90, today)
        st.plotly_chart(timeline_figure(m, sc, ev[ev.dog_id == dog_id], eps, chans, conds, today, chron, truth_mode),
                        width="stretch")
        al = D["alerts"]
        if len(al) and (al.dog_id == dog_id).any():
            st.markdown("**Alerts and why they fired**")
            st.dataframe(al[al.dog_id == dog_id].assign(condition=lambda x: x.condition.map(SHORT)).drop(columns="dog_id"),
                         hide_index=True, width="stretch")
        mt = D["maintenance"]
        if len(mt) and (mt.dog_id == dog_id).any():
            with st.expander(f"Device messages for this dog ({int((mt.dog_id == dog_id).sum())})"):
                st.dataframe(mt[mt.dog_id == dog_id].drop(columns="dog_id"), hide_index=True, width="stretch")
        if D["reference"] is not None:
            from praniiva.reference import REF_CHANNELS, percentile_vs_similar, LEARNING_DAYS
            recent = m[(m.day > today - 7) & (m.day <= today)]
            age_now = d.age0 + today / 365
            rows = []
            for ch in REF_CHANNELS:
                v = recent[ch].mean() if ch in recent else np.nan
                p = percentile_vs_similar(D["reference"], d.weight, age_now, ch, v)
                if p is not None:
                    rows.append(dict(measure=CH_LABEL.get(ch, ch), last_7_days=round(v, 1), percentile_vs_similar_dogs=round(p)))
            st.markdown(f"**Compared with similar dogs** (same size class and age band) - available from day 1, "
                        f"before the {LEARNING_DAYS}-day personal baseline is learned")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Orange x = day flagged as low data quality (loose collar, frozen sensor, glitch, wet fur) and ignored by the models. "
                   "Green band = this dog's normal from 2-4 months earlier (median ± 2 MAD), so slow declines show as leaving the band. Triangles = alerts "
                   "after debouncing; none fire after the vet's diagnosis (treatment tracking instead). Concern = model score / "
                   "alert threshold, from a model that never saw this dog - NOT a clinical probability.")

# ======================================================================================
elif page == "🧪 Live simulator":
    st.title("Live simulator - build a dog, inject a disease, watch the platform react")
    banner()
    c1, c2, c3 = st.columns(3)
    with c1:
        breed = st.selectbox("Breed", list(BREEDS), index=list(BREEDS).index("Labrador Retriever"))
        lo, hi = BREEDS[breed]["kg"]
        weight = st.slider("Weight (kg)", float(lo * 0.8), float(hi * 1.3), float((lo + hi) / 2))
        age = st.slider("Age (years)", 0.5, 15.0, 8.0)
    with c2:
        mmvd = st.checkbox("Vet-diagnosed MMVD (cardiac patient)")
        atopic = st.checkbox("Atopic (allergic skin)")
        otitis = st.checkbox("Otitis-prone")
        outdoor = st.checkbox("Mostly outdoors")
        compliance = st.slider("Owner collar compliance", 0.3, 1.0, 0.9)
    with c3:
        st.caption("Only the conditions injected here occur. The Atopic / Otitis boxes change the dog's baseline "
                   "scratching and head shaking; they do not cause flares on their own. Heat illness is simulated but "
                   "has no predictor yet.")
        inj = st.multiselect("Inject condition(s)", list(CONDITIONS) + ["heat_illness"], default=["chf_decomp"],
                             format_func=lambda c: SHORT.get(c, c))
        ev_day = st.slider("Event day (clinical recognition)", 80, SIM_DAYS - 20, 220)
        cfg = st.selectbox("Sensor configuration", list(SENSOR_CONFIGS), index=2,
                           format_func=lambda k: SENSOR_CONFIGS[k]["label"])
        seed = st.number_input("Random seed", 0, 9999, 7)
    if "chf_decomp" in inj and not mmvd:
        st.info("CHF monitoring is only offered to vet-diagnosed MMVD dogs (claim discipline), so this dog will be "
                "simulated as a diagnosed cardiac patient.")
        mmvd = True
    if st.button("▶ Run simulation", type="primary"):
        from praniiva.pipeline import make_custom_dog, simulate_single
        dog = make_custom_dog(breed, age, weight, mmvd, atopic, otitis, outdoor, compliance=compliance, seed=seed)
        with st.spinner("Simulating 400 days, measuring, scoring ..."):
            res = simulate_single(dog, {c: ev_day for c in inj}, cfg, MODELS, bundle(), seed=int(seed))
        st.session_state["live"] = res
    res = st.session_state.get("live")
    if res:
        sc, ev = res["scores"], res["events"]
        summary = []
        for _, e in ev.iterrows():
            c = e.condition
            if f"fired_{c}" not in sc:
                summary.append(dict(condition=SHORT.get(c, c), event_day=e.event_day, first_alert="(no model)", lead_days=None))
                continue
            from praniiva.params import positive_window
            a, b = positive_window(c, e.event_day, e.onset_day)
            hits = sc[(sc[f"fired_{c}"]) & (sc.day >= a) & (sc.day < b)].day
            summary.append(dict(condition=SHORT[c], event_day=e.event_day,
                                first_alert=int(hits.min()) if len(hits) else "missed",
                                lead_days=int(b - hits.min()) if len(hits) else None))
        fa = {SHORT[c]: int(sc[f"fired_{c}"].sum()) for c in CONDITIONS if f"fired_{c}" in sc}
        st.subheader("Did the platform see it coming?")
        st.dataframe(pd.DataFrame(summary), hide_index=True)
        st.caption(f"All alerts fired over 400 days (true + false): {fa}")
        stat = {SHORT[c]: sc[f"status_{c}"].value_counts().to_dict() for c in CONDITIONS if f"status_{c}" in sc}
        st.markdown("**Days the platform could / could not assess each condition** "
                    "(insufficient_data = key measurement missing, e.g. scratching while the collar was loose, "
                    "or the 35-day personal baseline not yet learned)")
        st.dataframe(pd.DataFrame(stat).T.fillna(0).astype(int), width="stretch")
        if "alerts" in res and len(res["alerts"]):
            st.markdown("**Each alert and why it fired**")
            st.dataframe(res["alerts"].assign(condition=res["alerts"].condition.map(SHORT)), hide_index=True, width="stretch")
        if "maintenance" in res and len(res["maintenance"]):
            st.markdown(f"**Device messages** ({len(res['maintenance'])}) - faults are reported to the owner, not turned into health alerts")
            st.dataframe(res["maintenance"].drop(columns="dog_id"), hide_index=True, width="stretch")
        conds = [c for c in inj if c in CONDITIONS] or ["oa_mobility"]
        chans = sorted({ch for c in conds for ch in COND_CH[c]}, key=list(CH_LABEL).index)[:5]
        st.plotly_chart(timeline_figure(res["meas"], sc, ev, res["episodes"], chans, conds), width="stretch")

# ======================================================================================
elif page == "📊 Algorithms & validation":
    st.title("Algorithms & validation")
    banner()
    rep = REPORTS / "ALGORITHM_BAKEOFF.md"
    tabs = st.tabs(["Verdicts", "Behaviour", "Detectors", "Risk models", "Sensor fusion", "Production models",
                    "Robustness (v0.2)"])
    with tabs[6]:
        f = REPORTS / "robustness.csv"
        if f.exists():
            rb = pd.read_csv(f)
            st.markdown("**Real-world faults (loose collar, frozen sensor, glitches, wet fur) - with and without the data-quality layer**")
            for metric, title in (("false_alerts_per_dog_month", "False alerts per dog-month (lower is better)"),
                                  ("sensitivity", "Events caught early (higher is better)")):
                piv = rb.pivot(index="condition", columns="scenario", values=metric)
                fig = go.Figure([go.Bar(name=c, x=piv.index, y=piv[c]) for c in piv.columns])
                st.plotly_chart(fig.update_layout(barmode="group", title=title, height=360), width="stretch")
            st.dataframe(rb, hide_index=True, width="stretch")
        for name, title in (("qc_detection.csv", "Fault detection by the QC layer"),
                            ("owner_feedback.csv", "Owner feedback loop (dismissed alerts relax that dog's threshold)")):
            f = REPORTS / name
            if f.exists():
                st.markdown(f"**{title}**")
                st.dataframe(pd.read_csv(f).round(3), hide_index=True, width="stretch")
    with tabs[0]:
        if rep.exists():
            txt = rep.read_text(encoding="utf-8")
            st.markdown(txt.split("## A.")[0])
        else:
            st.info("Run `python scripts/run_bakeoff.py` to generate the algorithm bake-off.")
    with tabs[1]:
        val = json.loads((MODELS / "behaviour_validation.json").read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("Random window split", f"{val['random_split']['accuracy']:.1%}")
        c2.metric("Unseen dogs", f"{val['leave_dogs_out']['accuracy']:.1%}")
        c3.metric("Unseen breed group", f"{val['leave_breed_group_out']['accuracy']:.1%}")
        lc = pd.DataFrame(val["learning_curve"])
        fig = go.Figure([go.Scatter(x=lc.train_dogs, y=lc.within_dog_acc, name="within training dogs (what small studies report)"),
                         go.Scatter(x=lc.train_dogs, y=lc.unseen_dog_acc, name="unseen dogs (honest)")])
        fig.update_layout(title="How many dogs must PRANIIVA recruit?", xaxis_title="dogs in training set",
                          yaxis_title="accuracy", height=380)
        st.plotly_chart(fig, width="stretch")
        if "sequence_study" in val:
            ss = val["sequence_study"]
            t = pd.DataFrame({b: dict(sens_raw=ss["raw"]["per_class"][b]["sensitivity"],
                                      sens_hmm=ss["hmm"]["per_class"][b]["sensitivity"],
                                      day_ppv_raw=ss["raw"]["per_class"][b]["day_level_ppv"],
                                      day_ppv_hmm=ss["hmm"]["per_class"][b]["day_level_ppv"]) for b in ss["raw"]["per_class"]}).T
            st.markdown("**Bout smoothing (sticky HMM) - precision at a real day's base rates, unseen dogs**")
            st.dataframe(t.style.format("{:.2f}"), width="stretch")
        if (REPORTS / "bakeoff_behaviour.csv").exists():
            st.dataframe(pd.read_csv(REPORTS / "bakeoff_behaviour.csv"), hide_index=True, width="stretch")
    with tabs[2]:
        if (REPORTS / "bakeoff_detectors.csv").exists():
            B = pd.read_csv(REPORTS / "bakeoff_detectors.csv")
            piv = B.pivot_table(index="detector", columns="condition", values="auroc")
            st.plotly_chart(go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="RdYlGn",
                                                 zmin=0.5, zmax=1, text=np.round(piv.values, 2), texttemplate="%{text}"))
                            .update_layout(title="Detector AUROC (single channel)", height=420), width="stretch")
            st.dataframe(B, hide_index=True, width="stretch")
    with tabs[3]:
        if (REPORTS / "bakeoff_risk.csv").exists():
            C = pd.read_csv(REPORTS / "bakeoff_risk.csv")
            piv = C.pivot_table(index="model", columns="condition", values="auprc")
            st.plotly_chart(go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Viridis",
                                                 text=np.round(piv.values, 2), texttemplate="%{text}"))
                            .update_layout(title="Risk-model AUPRC (dogs held out)", height=460), width="stretch")
            st.dataframe(C, hide_index=True, width="stretch")
        if (REPORTS / "split_inflation.csv").exists():
            st.markdown("**Random row split vs dogs held out; calibrated vs fixed 0.5 threshold**")
            st.dataframe(pd.read_csv(REPORTS / "split_inflation.csv"), hide_index=True, width="stretch")
    with tabs[4]:
        if (REPORTS / "sensor_ablation.csv").exists():
            A = pd.read_csv(REPORTS / "sensor_ablation.csv")
            piv = A.pivot_table(index="condition", columns="config", values="auprc")
            st.plotly_chart(go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Blues",
                                                 text=np.round(piv.values, 2), texttemplate="%{text}"))
                            .update_layout(title="AUPRC by sensor configuration", height=420), width="stretch")
            st.dataframe(A, hide_index=True, width="stretch")
    with tabs[5]:
        try:
            b = bundle()
            rows = []
            for c, v in b["conditions"].items():
                t = v.get("test_alert_metrics") or v.get("dev_alert_metrics") or {}
                ci = lambda k: (f"{t[k]:.2f} [{t[k + '_lo']:.2f}-{t[k + '_hi']:.2f}]"   # noqa: E731
                                if t and pd.notna(t.get(k)) else "n/a")
                rows.append(dict(condition=SHORT[c], model=v["model_name"],
                                 alerts="ON" if v.get("deploy_alerts") else "MONITOR ONLY",
                                 gate_basis=v.get("gate_basis", "?"), test_events=t.get("events"),
                                 sensitivity_95ci=ci("sensitivity"), alert_precision_95ci=ci("alert_precision"),
                                 false_alerts_per_dog_month_95ci=ci("false_alerts_per_dog_month"),
                                 calibration_ece=t.get("ece")))
            st.markdown("**Frozen models evaluated once on an untouched test cohort** (separate dogs, never used for "
                        "model choice, thresholds or calibration). 95% CIs by resampling dogs. Gate: precision >= 0.30, "
                        "sensitivity >= 0.30, >= 10 test events - a research filter, not a clinical standard.")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        except Exception as e:  # noqa: BLE001
            st.info(f"No production models yet ({e}).")

# ======================================================================================
elif page == "🔧 Device design":
    st.title("Device design - derived from the simulation")
    spec = ROOT / "docs" / "DEVICE_SPEC.md"
    st.markdown(spec.read_text(encoding="utf-8") if spec.exists() else "docs/DEVICE_SPEC.md not found")
    f = MODELS / "rr_validation_summary.csv"
    if f.exists():
        st.subheader("Resting-RR algorithm: sensor design sweep (simulated)")
        st.dataframe(pd.read_csv(f).round(2), hide_index=True, width="stretch")

# ======================================================================================
elif page == "📚 Parameters":
    st.title("Parameter registry")
    st.markdown("Every simulated quantity and where it comes from. Replace ASSUMPTION / SIM-ENRICHED values with "
                "PRANIIVA field data first.")
    st.subheader("Conditions")
    st.dataframe(pd.DataFrame([dict(key=k, **{kk: (", ".join(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})
                               for k, v in CONDITIONS.items()]), hide_index=True, width="stretch")
    st.subheader("Breeds")
    st.dataframe(pd.DataFrame(BREEDS).T, width="stretch")
    st.code((ROOT / "praniiva" / "params.py").read_text(encoding="utf-8"), language="python")
