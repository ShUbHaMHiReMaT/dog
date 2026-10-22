"""Check the last N seconds of the running recording: is the device still, and which face is up?"""
import glob, json, sys
import numpy as np, pandas as pd
sec = float(sys.argv[1]) if len(sys.argv) > 1 else 12
f = sorted(glob.glob("data/raw/MD_WRIST/*_raw.csv"))[-1]
r = pd.read_csv(f).tail(int(sec * 50))
A = r[["ax", "ay", "az"]].to_numpy()
sd = A.std(0) * 1000
m = A.mean(0)
ax = int(np.argmax(np.abs(m))); face = ("+" if m[ax] > 0 else "-") + "xyz"[ax]
still = bool((sd < 13).all())
print(json.dumps(dict(still=still, sd_mg=sd.round(2).tolist(), mean_g=m.round(4).tolist(), axis_up=face,
                      tilt_from_axis_deg=round(float(np.degrees(np.arccos(abs(m[ax]) / np.linalg.norm(m)))), 1),
                      t_end=int(r.timestamp_ms.iloc[-1]))))
