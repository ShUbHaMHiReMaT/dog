"""Worst-case scenarios (literature + real use) and the fit / placement check - a reduced run of the campaign
(tools/run_test_campaign.py families F and G). Every scenario must pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_test_campaign as campaign  # noqa: E402


def _run(fn, **kw):
    campaign.rows.clear()
    fn(**kw)
    bad = [r for r in campaign.rows if not r["passed"]]
    assert campaign.rows and not bad, bad[:5]


def test_worst_cases():
    _run(campaign.family_f, n=30, seed=101)


def test_fit_check():
    _run(campaign.family_g, n=12, seed=102)
