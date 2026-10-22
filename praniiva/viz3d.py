"""3D dog for the Live page: a three.js scene that follows the live activity label (praniiva/dog3d.html).

The live session writes static/live_pose.json every 2 s; the scene polls it (Streamlit static serving) and animates
smoothly between poses, so the page itself never has to re-render. three.js is vendored in static/vendor so the
page works without internet at the trial site.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSE_FILE = ROOT / "static" / "live_pose.json"


def dog3d_html(height: int = 430, mount: str = "chest", demo: str | None = None) -> str:
    from .live import ACTIVITIES
    tpl = (Path(__file__).parent / "dog3d.html").read_text(encoding="utf-8")
    three = (ROOT / "static" / "vendor" / "three.min.js").read_text(encoding="utf-8")
    cfg = json.dumps(dict(mount=mount, demo=demo, labels=ACTIVITIES))
    return (tpl.replace("__HEIGHT__", str(height)).replace("__CFG__", cfg)
            .replace("<script>__THREE__</script>", f"<script>{three}</script>"))
