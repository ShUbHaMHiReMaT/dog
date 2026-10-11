"""
Algorithm bake-off. Every candidate gets the same honest test (whole dogs held out), and is
ADOPTED only if it beats the simpler incumbent by a meaningful margin. Otherwise it is
REJECTED or kept as a BASELINE/COMPARATOR.

  Part A  behaviour recognition from raw IMU  (feature extraction + HistGB / XGBoost / RF / SVM /
          linear / stacked ensemble, vs a 1D CNN on raw signal)
  Part B  per-dog abnormality detectors (rolling-median z, ARIMA, Chronos, CUSUM change-point,
          linear-regression trend, deterministic rule)
  Part C  risk prediction (logistic/linear, HistGB, XGBoost, RF, Bayesian (naive-Bayes network),
          stacked ensemble, Cox survival, deterministic rules) + threshold calibration +
          precision/recall at event level
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import SVC

from .params import BEHAVIOURS, CONDITIONS

warnings.filterwarnings("ignore")
MARGIN = 0.01          # a more complex model must beat the incumbent by >= this to be adopted


# ======================================================================================
# Part A - behaviour recognition
# ======================================================================================
def _xgb_clf(n_classes):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=-1, verbosity=0)


def behaviour_candidates():
    return {
        "HistGB (incumbent)": lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                                   max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
        "XGBoost": lambda: _xgb_clf(len(BEHAVIOURS)),
        "Random Forest": lambda: RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "Linear (logistic)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "Stacked (HGB+RF+SVM -> LR)": lambda: StackingClassifier(
            estimators=[("hgb", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
                        ("rf", RandomForestClassifier(200, n_jobs=-1, random_state=0)),
                        ("svm", make_pipeline(StandardScaler(), SVC(C=10, probability=True)))],
            final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1),
    }


def _cnn_lodo(raw, y, dog, folds, epochs=12, seed=0):
    """1D CNN on raw signal. Input channels are rotation-invariant about the neck axis:
    |a|, a_x, sqrt(a_y^2+a_z^2)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    vm = np.linalg.norm(raw, axis=2)
    Xr = np.stack([vm, raw[:, :, 0], np.hypot(raw[:, :, 1], raw[:, :, 2])], axis=1).astype(np.float32)
    Xr = (Xr - Xr.mean((0, 2), keepdims=True)) / (Xr.std((0, 2), keepdims=True) + 1e-6)
    pred = np.full(len(y), -1)

    class Net(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(3, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, k))

        def forward(self, x):
            return self.h(self.f(x))

    for tr, te in folds:
        net = Net(len(BEHAVIOURS))
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
        Xt, yt = torch.tensor(Xr[tr]), torch.tensor(y[tr], dtype=torch.long)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(net(Xt[b]), yt[b])
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred[te] = net(torch.tensor(Xr[te])).argmax(1).numpy()
    return pred


def bakeoff_behaviour(out_dir: Path, n_dogs=48, per_class=24, verbose=True):
    from .behaviour import build_labelled_dataset
    ds = build_labelled_dataset(n_dogs, per_class, seed=11, keep_raw=True)
    X, y, dog = ds["X"], ds["y"], ds["dog"]
    folds = list(GroupKFold(5).split(X, y, dog))
    sc = BEHAVIOURS.index("scratching")
    rows = []

    def score(name, pred, secs):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(len(BEHAVIOURS)), zero_division=0)
        rows.append(dict(model=name, accuracy=float((pred == y).mean()), macro_f1=float(f1_score(y, pred, average="macro")),
                         scratch_sens=float(r[sc]), scratch_ppv=float(p[sc]), seconds=round(secs, 1)))
        if verbose:
            print(f"    {name:<30} acc {rows[-1]['accuracy']:.3f}  macroF1 {rows[-1]['macro_f1']:.3f}  "
                  f"scratch sens/PPV {r[sc]:.2f}/{p[sc]:.2f}  ({secs:.0f}s)")

    for name, mk in behaviour_candidates().items():
        t0 = time.time()
        pred = np.full(len(y), -1)
        for tr, te in folds:
            pred[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        score(name, pred, time.time() - t0)
    t0 = time.time()
    score("1D CNN on raw signal", _cnn_lodo(ds["raw"], y, dog, folds), time.time() - t0)
    df = pd.DataFrame(rows)
    inc = df.iloc[0]
    df["verdict"] = [
        "INCUMBENT" if i == 0 else
        ("ADOPT" if r.macro_f1 >= inc.macro_f1 + MARGIN else
         "TIE - keep simpler incumbent" if r.macro_f1 >= inc.macro_f1 - MARGIN else "REJECT")
        for i, r in df.iterrows()]
    df.to_csv(out_dir / "bakeoff_behaviour.csv", index=False)
    return df
