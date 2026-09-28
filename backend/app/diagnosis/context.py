"""Shared state for one diagnosis run, plus small helpers for evidence and charts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Ctx:
    model: dict[str, Any]
    preds: pd.DataFrame  # full log (baseline start .. window end)
    base: pd.DataFrame  # baseline window: 7 days before onset
    cur: pd.DataFrame  # analysis window: onset .. min(now, onset + 72h), keeps later incidents out
    cur_full: pd.DataFrame  # onset .. now, for blast-radius numbers
    onset: datetime
    window_end: datetime
    baseline_start: datetime
    data_end: datetime
    logs: list[dict[str, Any]]
    signals: dict[str, float] = field(default_factory=dict)
    signal_evidence: dict[str, list[str]] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def model_type(self) -> str:
        return self.model["model_type"]

    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.model["features"]]

    @property
    def decision_name(self) -> str:
        return self.model["decision_name"]

    def add_evidence(self, check: str, title: str, detail: str = "", chart: dict | None = None,
                     kind: str = "finding") -> str:
        ev_id = f"E{len(self.evidence) + 1}"
        self.evidence.append({"id": ev_id, "check": check, "title": title, "detail": detail,
                              "chart": chart, "kind": kind})
        return ev_id

    def signal(self, name: str, strength: float, evidence_ids: list[str] | None = None) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength >= self.signals.get(name, 0.0):
            self.signals[name] = round(strength, 3)
        self.signal_evidence.setdefault(name, [])
        for e in evidence_ids or []:
            if e not in self.signal_evidence[name]:
                self.signal_evidence[name].append(e)

    def add_check(self, name: str, label: str, status: str, headline: str) -> None:
        self.checks.append({"name": name, "label": label, "status": status, "headline": headline})

    def hourly(self, func, smooth: int = 1) -> tuple[list[str], list[float | None]]:
        """Apply func to each hour of the analysed span; returns ISO x values and y values."""
        span = self.preds
        hours = pd.date_range(pd.Timestamp(self.baseline_start).floor("h"), pd.Timestamp(self.window_end),
                              freq="h", inclusive="left")
        grouped = span.groupby(span["ts"].dt.floor("h"))
        values = grouped.apply(func) if len(span) else pd.Series(dtype=float)
        series = pd.Series(values, dtype=float).reindex(hours)
        if smooth > 1:
            series = series.rolling(smooth, min_periods=1).mean()
        return iso_list(hours), clean(series.to_numpy())


def iso(ts: datetime | pd.Timestamp | None) -> str | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M")


def iso_list(idx) -> list[str]:
    return [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M") for t in idx]


def clean(values, digits: int = 4) -> list[float | None]:
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), digits)
            for v in np.asarray(values, dtype=float)]


def line_chart(title: str, x: list[str], series: dict[str, list], y_format: str, onset: datetime,
               threshold: float | None = None, threshold_label: str = "") -> dict:
    return {"type": "line", "title": title, "x": x,
            "series": [{"name": k, "y": v} for k, v in series.items()],
            "y_format": y_format, "markers": [{"x": iso(onset), "label": "onset"}],
            "threshold": threshold, "threshold_label": threshold_label}


def bar_chart(title: str, categories: list[str], series: dict[str, list], y_format: str,
              threshold: float | None = None, threshold_label: str = "") -> dict:
    return {"type": "bar", "title": title, "categories": categories,
            "series": [{"name": k, "values": clean(v)} for k, v in series.items()],
            "y_format": y_format, "threshold": threshold, "threshold_label": threshold_label}
