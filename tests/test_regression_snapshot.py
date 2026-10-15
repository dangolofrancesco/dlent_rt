"""Stage-C gate in the fast suite: Phase 0 + online run on the synthetic config are BIT-IDENTICAL to
tests/data/snapshot_synthetic.npz (python -m tests.regression_snapshot save synthetic only on an
intended, approved numeric change)."""
import numpy as np

from tests.regression_snapshot import compare, path, snapshot


def test_regression_snapshot_synthetic():
    diff = compare(snapshot("synthetic"), dict(np.load(path("synthetic"))))
    assert diff == [], "snapshot differs:\n" + "\n".join(diff)
