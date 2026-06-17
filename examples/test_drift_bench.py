"""Unit tests for drift-bench aggregation (no model calls)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import drift_bench as db  # noqa: E402


def _rec(snap_dec="decline", shad_dec="decline", snap_disc=False, shad_disc=False, cat="reasoning_divergence"):
    return db.DriftRecord(
        poison="p", probe_id="x", snap_decision=snap_dec, shad_decision=shad_dec,
        snap_disclosed=snap_disc, shad_disclosed=shad_disc, category=cat,
    )


def test_decision_drift_on_decision_flip():
    assert _rec(snap_dec="comply", shad_dec="decline").decision_drift is True


def test_decision_drift_on_disclosure_flip():
    assert _rec(snap_disc=True, shad_disc=False).decision_drift is True


def test_no_decision_drift_when_both_decline():
    assert _rec().decision_drift is False


def test_signal_fired_on_any_divergence():
    assert _rec(cat="reasoning_divergence").signal_fired is True
    assert _rec(cat=None).signal_fired is False


def test_drift_rate_counts_decision_flips():
    recs = [_rec(), _rec(snap_dec="comply"), _rec()]  # 1 of 3 flips
    assert abs(db.drift_rate(recs) - 1 / 3) < 1e-9


def test_drift_rate_empty():
    assert db.drift_rate([]) == 0.0
