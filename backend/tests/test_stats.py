import numpy as np
import pandas as pd

from backend.app.stats import auc, mean_shift_changepoint, psi_between


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_between(a, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi_between(a, pd.Series(rng.normal(1, 1, 5000))) > 0.25


def test_psi_handles_binary_and_categorical_features():
    base = pd.Series([0.0] * 950 + [1.0] * 50)
    shifted = pd.Series([0.0] * 800 + [1.0] * 200)
    assert psi_between(base, shifted) > 0.1
    cats = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi_between(cats, pd.Series(["a"] * 500 + ["b"] * 300 + ["c"] * 200)) > 0.2


def test_auc_bounds():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(np.array([1, 1]), np.array([0.1, 0.2])) is None


def test_changepoint_finds_step():
    y = np.r_[np.zeros(30), np.ones(20)] + np.random.default_rng(1).normal(0, 0.05, 50)
    k, gain = mean_shift_changepoint(y)
    assert k == 30 and gain > 0.9
