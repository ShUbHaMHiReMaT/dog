"""Live activity / posture rules must work for ANY mounting orientation of the device (chest, neck, back)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from praniiva.live import _unit, window_activity

T = np.arange(100) / 50


def _gravity(pitch=0.0, roll=0.0):
    """Accelerometer reading at rest (+1 g up) in the dog's body frame; pitch > 0 = trunk nose-up (sitting)."""
    return Rot.from_euler("yx", [-pitch, roll], degrees=True).inv().apply([0, 0, 1])


def _win(rng, mount, pitch=0.0, roll=0.0):
    return mount.inv().apply(_gravity(pitch, roll)) + rng.normal(0, 0.002, (100, 3))


def test_posture_any_mounting():
    rng = np.random.default_rng(0)
    for k in range(100):
        m = Rot.random(random_state=k)
        sit_p = rng.uniform(30, 70)
        stand, sit = _unit(_win(rng, m).mean(0)), _unit(_win(rng, m, sit_p).mean(0))
        cal = dict(stand=stand, sit=sit, sit_pitch=float(np.degrees(np.arccos(stand @ sit))))
        cases = [("standing", dict(pitch=rng.uniform(-10, 10), roll=rng.uniform(-10, 10)), 0),
                 ("standing", dict(pitch=-25), 0),
                 ("sitting", dict(pitch=sit_p + rng.uniform(-10, 10)), 0),
                 ("lying_side", dict(roll=rng.choice([-1, 1]) * rng.uniform(70, 100)), 0),
                 ("lying_side", dict(roll=180), 0),
                 ("lying_belly", dict(pitch=rng.uniform(-8, 8)), 130)]
        for want, kw, still_s in cases:
            assert window_activity(_win(rng, m, **kw), cal, still_s)[0] == want, (k, want, kw)


def test_motion_any_mounting():
    rng = np.random.default_rng(1)
    hits = {}
    for k in range(100):
        g = Rot.random(random_state=k).inv().apply([0, 0, 1])
        u1, u2 = _unit(rng.normal(size=3)), _unit(rng.normal(size=3))
        for want, f, a in [("walking", rng.uniform(1.5, 2.5), rng.uniform(0.08, 0.25)),
                           ("running", rng.uniform(2.5, 4), rng.uniform(0.6, 1.2)),
                           ("shaking", rng.uniform(5, 8), rng.uniform(0.3, 1.5)),
                           ("still", 1, 0.0)]:
            x = (g + np.outer(a * np.sin(2 * np.pi * f * T), u1) + np.outer(0.3 * a * np.sin(4 * np.pi * f * T + 1), u2)
                 + rng.normal(0, 0.002, (100, 3)))
            hits.setdefault(want, []).append(window_activity(x, None)[0] == want)
    for want, h in hits.items():
        assert np.mean(h) >= 0.95, (want, np.mean(h))
