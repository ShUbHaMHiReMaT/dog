"""
Tier 1 validation study: behaviour classification + resting-RR algorithm.

Reproduces the dossier's validation discipline:
  * random window split (optimistic, what most papers report)
  * leave-one-dog-out style GroupKFold (honest)
  * leave-one-breed-group-out (generalisation to unseen breed types)

The LODO confusion matrix and RR error profile are saved and become the measurement-error
model used by the Tier 2 (daily) simulator - so the daily data is only as good as the
behaviour model actually is.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .imu import FEATURE_NAMES, dog_motion_profile, estimate_rr, features_matrix, synth_segment
from .params import BEHAVIOURS, BREEDS, DEFAULT_SENSOR


def _clf():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def build_labelled_dataset(n_dogs=60, per_class=50, seed=1, sensor=None, keep_raw=False):
    """Simulated labelled sessions (dossier Phase 2: 60-100 dogs, 7+ breeds)."""
    rng = np.random.default_rng(seed)
    names = list(BREEDS)
    X, y, dog, breed, group = [], [], [], [], []
    raw = []
    dogs_meta = []
    for d in range(n_dogs):
        b = names[d % len(names)]
        info = BREEDS[b]
        w = rng.uniform(*info["kg"])
        prof = dog_motion_profile(w, info["coat"], rng, info["cfr"])
        lame = rng.choice([0.0, 0.0, 0.0, rng.uniform(0.2, 0.6)])
        dogs_meta.append(dict(dog=d, breed=b, weight=w, lameness=lame))
        for c, beh in enumerate(BEHAVIOURS):
            n = per_class if beh not in ("head_shaking",) else per_class // 2
            ws = [synth_segment(beh, prof, rng, sensor=sensor,
                                rr_bpm=rng.uniform(12, 24),
                                lameness=lame if beh in ("walking", "trotting") else 0.0)
                  for _ in range(n)]
            X.append(features_matrix(np.stack(ws)))
            if keep_raw:
                raw.append(np.stack(ws).astype(np.float32))
            y += [c] * n
            dog += [d] * n
            breed += [b] * n
            group += [info["group"]] * n
    X = np.vstack(X)
    y = np.array(y)
    # 2% annotation noise (dual-observer agreement is never perfect)
    flip = rng.random(len(y)) < 0.02
    y_noisy = y.copy()
    y_noisy[flip] = rng.integers(0, len(BEHAVIOURS), flip.sum())
    return dict(X=X, y=y_noisy, y_true=y, dog=np.array(dog), breed=np.array(breed),
                group=np.array(group), meta=pd.DataFrame(dogs_meta),
                raw=np.concatenate(raw) if keep_raw else None)


def _cv_predict(X, y, splits):
    pred = np.full(len(y), -1)
    for tr, te in splits:
        m = _clf().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _summary(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(BEHAVIOURS)), zero_division=0)
    return dict(accuracy=float((y == pred).mean()), macro_f1=float(f1_score(y, pred, average="macro")),
                per_class={b: dict(sensitivity=float(r[i]), ppv=float(p[i]), f1=float(f[i]))
                           for i, b in enumerate(BEHAVIOURS)})


def run_behaviour_study(out_dir: Path, n_dogs=60, per_class=30, seed=1, verbose=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = build_labelled_dataset(n_dogs, per_class, seed)
    X, y, dog, grp = ds["X"], ds["y"], ds["dog"], ds["group"]
    res = {}
    rnd = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    res["random_split"] = _summary(y, _cv_predict(X, y, rnd))
    lodo_pred = _cv_predict(X, y, list(GroupKFold(5).split(X, y, dog)))
    res["leave_dogs_out"] = _summary(y, lodo_pred)
    groups = np.unique(grp)
    lbo = [(np.where(grp != g)[0], np.where(grp == g)[0]) for g in groups]
    lbo_pred = _cv_predict(X, y, lbo)
    res["leave_breed_group_out"] = _summary(y, lbo_pred)
    res["per_breed_group_lbo_accuracy"] = {g: float((y[grp == g] == lbo_pred[grp == g]).mean()) for g in groups}
    # learning curve: how many recruited dogs before unseen-dog accuracy stops improving?
    rng = np.random.default_rng(seed)
    dogs = np.unique(dog)
    curve = []
    for rep in range(3):
        perm = rng.permutation(dogs)
        test_d, pool = perm[:12], perm[12:]
        te = np.isin(dog, test_d)
        for k in (2, 4, 8, 16, 32, len(pool)):
            tr = np.isin(dog, pool[:k])
            m = _clf().fit(X[tr], y[tr])
            acc_unseen = float((m.predict(X[te]) == y[te]).mean())
            # 'within-dog' accuracy: random split inside the training dogs (what small studies report)
            idx = np.where(tr)[0]
            rng.shuffle(idx)
            h = len(idx) // 5
            m2 = _clf().fit(X[idx[h:]], y[idx[h:]])
            acc_within = float((m2.predict(X[idx[:h]]) == y[idx[:h]]).mean())
            curve.append(dict(train_dogs=int(k), rep=rep, unseen_dog_acc=acc_unseen, within_dog_acc=acc_within))
    res["learning_curve"] = pd.DataFrame(curve).groupby("train_dogs")[["unseen_dog_acc", "within_dog_acc"]].mean().reset_index().to_dict("records")
    cm = confusion_matrix(ds['y_true'], lodo_pred, labels=range(len(BEHAVIOURS))).astype(float)  # vs true labels
    cm_norm = cm / cm.sum(1, keepdims=True)
    res["confusion_lodo_row_normalised"] = cm_norm.tolist()
    res["n_dogs"] = n_dogs
    res["n_windows"] = int(len(y))
    final = _clf().fit(X, y)
    joblib.dump(dict(model=final, behaviours=BEHAVIOURS, features=FEATURE_NAMES), out_dir / "behaviour_model.joblib")
    (out_dir / "behaviour_validation.json").write_text(json.dumps(res, indent=2))
    if verbose:
        print(f"  behaviour accuracy  random-split {res['random_split']['accuracy']:.3f} | "
              f"leave-dogs-out {res['leave_dogs_out']['accuracy']:.3f} | "
              f"leave-breed-group-out {res['leave_breed_group_out']['accuracy']:.3f}")
        for r in res["learning_curve"]:
            print(f"    train dogs {r['train_dogs']:>3}: within-dog {r['within_dog_acc']:.3f}  unseen-dog {r['unseen_dog_acc']:.3f}")
    return res


# --------------------------------------------------------------------------------------
# Resting RR validation + sensor design sweep
# --------------------------------------------------------------------------------------
def run_rr_study(out_dir: Path, n=120, seed=2, verbose=True):
    rng = np.random.default_rng(seed)
    configs = {
        "16-bit, 150 ug/rtHz (recommended)": dict(bits=16, noise_ug_rthz=150),
        "12-bit, 150 ug/rtHz": dict(bits=12, noise_ug_rthz=150),
        "16-bit, 400 ug/rtHz (cheap MEMS)": dict(bits=16, noise_ug_rthz=400),
        "16-bit, 150 ug/rtHz, 25 Hz": dict(bits=16, noise_ug_rthz=150, fs=25),
    }
    rows = []
    for cname, cfg in configs.items():
        sensor = dict(DEFAULT_SENSOR, **cfg)
        for coat in ("short", "medium", "long"):
            for i in range(n // 3):
                w = rng.uniform(3, 45)
                prof = dog_motion_profile(w, coat, rng)
                rr = rng.uniform(10, 40)
                seg = synth_segment("lying_rest", prof, rng, seconds=60, rr_bpm=rr, sensor=sensor)
                est, q = estimate_rr(seg, fs=sensor["fs"])
                rows.append(dict(config=cname, coat=coat, weight=w, rr_true=rr, rr_est=est, quality=q))
    df = pd.DataFrame(rows)
    df["err"] = df.rr_est - df.rr_true
    summ = (df.groupby(["config", "coat"])
              .apply(lambda g: pd.Series(dict(
                  yield_=g.rr_est.notna().mean(),
                  bias=g.err.mean(),
                  loa_lo=g.err.mean() - 1.96 * g.err.std(),
                  loa_hi=g.err.mean() + 1.96 * g.err.std(),
                  within_2bpm=(g.err.abs() <= 2).sum() / max(g.rr_est.notna().sum(), 1))), include_groups=False)
              .reset_index())
    df.to_csv(out_dir / "rr_validation_raw.csv", index=False)
    summ.to_csv(out_dir / "rr_validation_summary.csv", index=False)
    rec = df[df.config.str.contains("recommended")]
    profile = {}
    for coat, g in rec.groupby("coat"):
        e = g.err.dropna()
        profile[coat] = dict(yield_=float(g.rr_est.notna().mean()), bias=float(e.mean()), sd=float(e.std()))
    (out_dir / "rr_error_profile.json").write_text(json.dumps(profile, indent=2))
    if verbose:
        print(summ.round(2).to_string(index=False))
    return summ, profile


# --------------------------------------------------------------------------------------
# Temporal post-processing: sticky HMM (Viterbi) with day-budget priors.
# Behaviour comes in bouts; per-window classifiers do not know that, and at real base rates
# (a dog lies ~1300 min/day) per-window false positives swamp rare behaviours like scratching.
# --------------------------------------------------------------------------------------
BOUT_WINDOWS = dict(lying_rest=(60, 600), standing=(5, 60), walking=(15, 300), trotting=(5, 60),
                    scratching=(2, 6), head_shaking=(1, 1), eating=(30, 200), drinking=(5, 30), panting=(15, 150))
DAY_MINUTES = dict(lying_rest=1290, standing=60, walking=30, trotting=10, scratching=1.5, head_shaking=0.15,
                   eating=8, drinking=3.5, panting=25)


def viterbi_smooth(proba: np.ndarray, train_prior: np.ndarray | None = None) -> np.ndarray:
    """proba: (T, K) classifier probabilities for consecutive 2 s windows. Returns smoothed labels."""
    K = proba.shape[1]
    prior = np.array([DAY_MINUTES[b] for b in BEHAVIOURS]); prior = prior / prior.sum()
    train_prior = np.full(K, 1 / K) if train_prior is None else train_prior
    mean_len = np.array([np.mean(BOUT_WINDOWS[b]) for b in BEHAVIOURS])
    stay = 1 - 1 / np.maximum(mean_len, 1.5)
    A = np.empty((K, K))
    for i in range(K):
        A[i] = (1 - stay[i]) * prior / (1 - prior[i]) * (1 - np.eye(K)[i])
        A[i, i] = stay[i]
    logA = np.log(A / A.sum(1, keepdims=True))
    # convert posteriors (trained on balanced classes) into scaled likelihoods
    logE = np.log(np.clip(proba, 1e-6, 1)) - np.log(train_prior)
    T = len(proba)
    dp = np.log(prior) + logE[0]
    back = np.zeros((T, K), dtype=np.int16)
    for t in range(1, T):
        cand = dp[:, None] + logA
        back[t] = cand.argmax(0)
        dp = cand.max(0) + logE[t]
    path = np.empty(T, dtype=int)
    path[-1] = dp.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    # head shakes are ~1 s events, not bouts: they bypass the smoother (event detector)
    hs = BEHAVIOURS.index("head_shaking")
    path[(proba.argmax(1) == hs) & (proba[:, hs] > 0.85)] = hs
    return path


def simulate_sequence(prof, rng, n_windows=1500, sensor=None, rr=16.0):
    """Continuous bout-structured stream cut into 2 s windows (label = majority), with one collar
    angle per session - boundary windows are mixed exactly as on real hardware. Class time-shares
    are boosted vs a real day so rare classes get enough windows; day-level PPV is recomputed at
    real proportions afterwards."""
    sensor = sensor or DEFAULT_SENSOR
    fs = sensor["fs"]
    wl = int(2 * fs)
    weights = np.array([3, 2, 2, 1, 2, 1, 2, 1, 2], float)
    angle = rng.uniform(-np.pi, np.pi)
    sig, lab = [], []
    total = 0
    while total < n_windows * wl:
        b = rng.choice(len(BEHAVIOURS), p=weights / weights.sum())
        lo, hi = BOUT_WINDOWS[BEHAVIOURS[b]]
        L = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        L = min(L, 80)
        secs = L * 2 * rng.uniform(0.8, 1.2)
        seg = synth_segment(BEHAVIOURS[b], prof, rng, seconds=secs, rr_bpm=rr, sensor=sensor,
                            collar_angle=angle + rng.normal(0, 0.15), transitions=False)
        sig.append(seg); lab.append(np.full(len(seg), b))
        total += len(seg)
    sig = np.concatenate(sig)[: n_windows * wl]
    lab = np.concatenate(lab)[: n_windows * wl]
    W = sig.reshape(n_windows, wl, 3)
    L = lab.reshape(n_windows, wl)
    labels = np.array([np.bincount(r, minlength=len(BEHAVIOURS)).argmax() for r in L])
    return W, labels


def run_sequence_study(out_dir: Path, n_test_dogs=20, seed=5, verbose=True):
    bundle = joblib.load(out_dir / "behaviour_model.joblib")
    model = bundle["model"]
    rng = np.random.default_rng(seed)
    names = list(BREEDS)
    cm_raw = np.zeros((len(BEHAVIOURS),) * 2); cm_hmm = np.zeros_like(cm_raw)
    for d in range(n_test_dogs):                                   # NEW dogs the model has never seen
        info = BREEDS[names[(d * 3) % len(names)]]
        prof = dog_motion_profile(rng.uniform(*info["kg"]), info["coat"], rng, info["cfr"])
        W, lab = simulate_sequence(prof, rng)
        P = model.predict_proba(features_matrix(W))
        raw = P.argmax(1)
        sm = viterbi_smooth(P)
        cm_raw += confusion_matrix(lab, raw, labels=range(len(BEHAVIOURS)))
        cm_hmm += confusion_matrix(lab, sm, labels=range(len(BEHAVIOURS)))
    res = {}
    day = np.array([DAY_MINUTES[b] for b in BEHAVIOURS])
    for name, cm in (("raw", cm_raw), ("hmm", cm_hmm)):
        cmn = cm / cm.sum(1, keepdims=True)
        measured = day @ cmn
        res[name] = dict(
            confusion=cmn.tolist(),
            per_class={b: dict(sensitivity=float(cmn[i, i]),
                               day_level_ppv=float(day[i] * cmn[i, i] / measured[i]),
                               measured_vs_true_minutes=float(measured[i] / day[i]))
                       for i, b in enumerate(BEHAVIOURS)})
    val = json.loads((out_dir / "behaviour_validation.json").read_text())
    val["sequence_study"] = res
    val["confusion_for_measurement"] = res["hmm"]["confusion"]
    (out_dir / "behaviour_validation.json").write_text(json.dumps(val, indent=2))
    if verbose:
        print("    class          sens raw/HMM   day-level PPV raw/HMM   measured/true minutes raw/HMM")
        for b in BEHAVIOURS:
            r, h = res["raw"]["per_class"][b], res["hmm"]["per_class"][b]
            print(f"    {b:<14} {r['sensitivity']:.2f}/{h['sensitivity']:.2f}        {r['day_level_ppv']:.2f}/{h['day_level_ppv']:.2f}"
                  f"                {r['measured_vs_true_minutes']:.1f}/{h['measured_vs_true_minutes']:.1f}")
    return res
