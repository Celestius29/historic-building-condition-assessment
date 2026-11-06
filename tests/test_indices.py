import numpy as np
import pytest
from morphology.indices import crack_indices


def _line():
    m = np.zeros((200, 300), bool); m[98:103, 20:280] = True
    return m


def test_single_line():
    r = crack_indices(_line())
    assert r["length_px"] == pytest.approx(256.4, abs=3)
    assert r["mean_width_px"] == pytest.approx(5, abs=0.2)
    assert r["tortuosity"] == pytest.approx(1.0, abs=0.02)
    assert r["n_branches"] == 1
    assert r["n_junctions"] == 0


def test_line_with_vertical_bar():
    m = _line(); m[20:180, 148:153] = True
    r = crack_indices(m)
    assert r["n_junctions"] == 1
    assert r["n_branches"] == 4
