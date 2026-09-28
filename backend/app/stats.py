"""Small, dependency-light statistics used by monitoring and diagnosis."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def psi_from_props(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two binned distributions."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_edges(values: np.ndarray, bins: int = 10) -> list[float]:
    """Quantile bin edges (open-ended at both sides) computed on a reference sample."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    qs = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    return [-np.inf, *qs.tolist(), np.inf]


def bin_numeric(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin index per value; NaN -> -1."""
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges[1:-1]), v, side="left")  # bins are (lo, hi]
    idx[np.isnan(v)] = -1
    return idx


def bin_categorical(values: pd.Series, categories: list[str]) -> np.ndarray:
    """Category index per value; unseen categories go to an extra 'other' bin; NaN -> -1."""
    lookup = {c: i for i, c in enumerate(categories)}
    other = len(categories)
    out = np.array([lookup.get(x, other) if isinstance(x, str) else -1 for x in values], dtype=int)
    return out


def props(idx: np.ndarray, n_bins: int) -> np.ndarray:
    idx = idx[idx >= 0]
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    total = counts.sum()
    return counts / total if total else counts


def psi_between(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI between two raw samples, binning on the baseline. NaNs are ignored."""
    b = baseline.dropna()
    c = current.dropna()
    if len(b) < 20 or len(c) < 20:
        return 0.0
    if pd.api.types.is_numeric_dtype(b):
        edges = numeric_edges(b.to_numpy(dtype=float), bins)
        n = len(edges) - 1
        return psi_from_props(props(bin_numeric(b.to_numpy(float), edges), n), props(bin_numeric(c.to_numpy(float), edges), n))
    cats = sorted(set(b.astype(str)) | set(c.astype(str)))
    n = len(cats) + 1
    return psi_from_props(props(bin_categorical(b.astype(str), cats), n), props(bin_categorical(c.astype(str), cats), n))


def auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    """ROC AUC via the Mann-Whitney rank statistic."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = ~np.isnan(s)
    y, s = y[mask], s[mask]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mean_shift_changepoint(y: np.ndarray, min_size: int = 3) -> tuple[int | None, float]:
    """Best single split of a series into two constant-mean segments.

    Returns (index of first point after the change, relative SSE reduction).
    """
    y = np.asarray(y, dtype=float)
    if np.isnan(y).all():
        return None, 0.0
    y = np.where(np.isnan(y), np.nanmean(y), y)
    n = len(y)
    if n < 2 * min_size + 1:
        return None, 0.0
    total_sse = float(((y - y.mean()) ** 2).sum())
    if total_sse == 0:
        return None, 0.0
    cs = np.cumsum(y)
    cs2 = np.cumsum(y**2)
    best_k, best_sse = None, total_sse
    for k in range(min_size, n - min_size + 1):
        left_n, right_n = k, n - k
        left_sum, right_sum = cs[k - 1], cs[-1] - cs[k - 1]
        left_sse = cs2[k - 1] - left_sum**2 / left_n
        right_sse = (cs2[-1] - cs2[k - 1]) - right_sum**2 / right_n
        sse = left_sse + right_sse
        if sse < best_sse:
            best_k, best_sse = k, sse
    return best_k, float(1 - best_sse / total_sse)


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def pp(x: float, digits: int = 1) -> str:
    """Percentage-point delta with sign."""
    return f"{x * 100:+.{digits}f}pp"
