"""Shared look & feel: palette, CSS, badges, KPI tiles and Plotly chart builders.

Palette follows a validated categorical order (blue, orange, aqua), a one-hue sequential
ramp for magnitude, and reserved status colours that always ship with an icon + label.
"""
from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# ---- palette ------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASELINE = "#c3c2b7"  # de-emphasis for 'before' / non-selected marks
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SEVERITY = {  # alert severity -> (status colour, icon, label)
    "critical": (STATUS["critical"], "▲", "Critical"),
    "high": (STATUS["serious"], "▲", "High"),
    "medium": (STATUS["warning"], "●", "Medium"),
    "low": (STATUS["good"], "●", "Low"),
}
SEV_LEVEL = {"SEV1": "critical", "SEV2": "high", "SEV3": "medium"}
INC_STATUS = {
    "open": ("#c3c2b7", "Open"), "investigating": (SERIES[0], "Investigating"),
    "mitigated": (STATUS["warning"], "Mitigated"), "resolved": (STATUS["good"], "Resolved"),
}
HEALTH = {"critical": (STATUS["critical"], "▲", "Critical"), "degraded": (STATUS["serious"], "▲", "Degraded"),
          "healthy": (STATUS["good"], "✓", "Healthy")}

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --grid:{GRID}; --surface:{SURFACE}; }}
html, body, [class*="css"] {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.pill {{ display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
        font-size:12.5px; font-weight:600; color:var(--ink); background:#fff;
        border:1px solid rgba(11,11,11,.10); white-space:nowrap; line-height:20px; }}
.pill .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.pill .ico {{ font-size:10px; }}
.muted {{ color: var(--muted); }}
.sub {{ color: var(--ink2); font-size: 0.92rem; }}
.kicker {{ text-transform: uppercase; letter-spacing: .06em; font-size: 11.5px; color: var(--muted); font-weight: 600; }}
.hyp-title {{ font-size: 1.35rem; font-weight: 650; margin: 2px 0 4px 0; }}
.hero {{ font-size: 3rem; font-weight: 650; line-height: 1; }}
.ev-title {{ font-weight: 600; margin-bottom: 2px; }}
.ev-detail {{ color: var(--ink2); font-size: .9rem; }}
.contrib {{ font-variant-numeric: tabular-nums; font-size: 12px; color: var(--ink2); background:#f0efec;
           border-radius: 6px; padding: 1px 6px; margin-left: 6px; }}
.stepper {{ display:flex; gap:0; margin: 6px 0 2px 0; }}
.step {{ flex:1; text-align:center; font-size:12.5px; font-weight:600; color: var(--muted); padding: 8px 0 6px 0;
        border-bottom: 3px solid var(--grid); }}
.step.done {{ color: var(--ink); border-bottom-color: {SERIES[0]}; }}
.step.now {{ color: var(--ink); border-bottom-color: {INK}; }}
.tl-row {{ display:flex; gap:12px; padding:6px 0; border-bottom:1px solid var(--grid); font-size:.9rem; }}
.tl-time {{ color: var(--muted); min-width: 118px; font-variant-numeric: tabular-nums; }}
.tl-actor {{ color: var(--ink2); min-width: 150px; }}
.checks {{ display:flex; flex-wrap:wrap; gap:8px; }}
.check {{ border:1px solid rgba(11,11,11,.10); border-radius:10px; padding:6px 10px; background:#fff; font-size:12.5px; }}
.check b {{ display:block; font-size:12.5px; }}
div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def pill(label: str, color: str, icon: str | None = None) -> str:
    mark = f'<span class="ico" style="color:{color}">{icon}</span>' if icon else f'<span class="dot" style="background:{color}"></span>'
    return f'<span class="pill">{mark}{esc(label)}</span>'


def severity_pill(sev: str) -> str:
    if sev in SEV_LEVEL:
        color, icon, _ = SEVERITY[SEV_LEVEL[sev]]
        return pill(sev, color, icon)
    color, icon, label = SEVERITY.get(sev, (MUTED, "●", sev))
    return pill(label, color, icon)


def status_pill(status: str) -> str:
    color, label = INC_STATUS.get(status, (MUTED, status))
    return pill(label, color)


def health_pill(h: str) -> str:
    color, icon, label = HEALTH[h]
    return pill(label, color, icon)


def fmt_dt(iso: str | None, with_day: bool = True) -> str:
    if not iso:
        return "—"
    d = datetime.fromisoformat(iso)
    return d.strftime("%a %d %b %H:%M" if with_day else "%H:%M")


def fmt_age(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.0f} h"
    return f"{hours / 24:.1f} d"


def pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def compact(n: float | None) -> str:
    if n is None:
        return "—"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs(n) >= 1e4:
        return f"{n / 1e3:.0f}K"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}"


# ---- charts -------------------------------------------------------------------------

_TICK = {"pct": ".0%", "ms": ",.0f", "num": ".2f", "int": ",.0f"}
_HOVER = {"pct": ".1%", "ms": ",.0f", "num": ".3f", "int": ",.0f"}
_SUFFIX = {"ms": " ms"}


def _base(fig: go.Figure, height: int, title: str | None, y_format: str, legend: bool) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=34 if title else 8, b=8),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_2, size=12),
        title=dict(text=title, font=dict(size=13.5, color=INK), x=0, xanchor="left", y=0.98) if title else None,
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                                       font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(color=INK, size=12)),
        barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     tickformat=_TICK.get(y_format, ""), ticksuffix=_SUFFIX.get(y_format, ""), rangemode="tozero")
    return fig


def line_chart(x: list, series: dict[str, list], y_format: str = "num", title: str | None = None, height: int = 260,
               bands: list[dict] | None = None, markers: list[dict] | None = None, threshold: float | None = None,
               threshold_label: str = "", emphasis_first: bool = False) -> go.Figure:
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        color = SERIES[i % len(SERIES)] if not emphasis_first or i == 0 else BASELINE
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=f"%{{y:{_HOVER.get(y_format, '')}}}{_SUFFIX.get(y_format, '')}<extra>{esc(name)}</extra>",
                                 connectgaps=False))
    for b in bands or []:
        fig.add_vrect(x0=b["x0"], x1=b["x1"], fillcolor=b.get("color", STATUS["serious"]), opacity=0.09, line_width=0,
                      layer="below")
        if b.get("label"):
            fig.add_annotation(x=b["x0"], y=1, yref="paper", text=esc(b["label"]), showarrow=False, xanchor="left",
                               yanchor="top", font=dict(size=11, color=INK_2), bgcolor="rgba(252,252,251,.85)")
    for m in markers or []:
        fig.add_shape(type="line", x0=m["x"], x1=m["x"], y0=0, y1=1, yref="paper",
                      line=dict(color=AXIS, width=1))
        fig.add_annotation(x=m["x"], y=1, yref="paper", text=esc(m.get("label", "")), showarrow=False,
                           xanchor="left", yanchor="top", xshift=3, font=dict(size=11, color=INK_2),
                           bgcolor="rgba(252,252,251,.85)")
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=MUTED, width=1))
        if threshold_label:
            fig.add_annotation(x=1, xref="paper", y=threshold, text=esc(threshold_label), showarrow=False,
                               xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=AXIS, spikedash="solid")
    fig = _base(fig, height, title, y_format, legend=len(series) > 1)
    ymax = max((v for y in series.values() for v in y if v is not None), default=0)
    if y_format == "pct" and ymax < 0.05:
        fig.update_yaxes(tickformat=".1%")
    return fig


def bar_chart(categories: list, series: dict[str, list], y_format: str = "num", title: str | None = None,
              height: int = 240, threshold: float | None = None, threshold_label: str = "",
              horizontal: bool = False, highlight: int | None = None, text: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    names = list(series)
    for i, name in enumerate(names):
        vals = series[name]
        if len(names) == 2:  # baseline vs incident: de-emphasise the reference
            color = BASELINE if i == 0 else SERIES[0]
        else:
            color = SERIES[0]
        if highlight is not None and len(names) == 1:
            color = [SERIES[0] if j == highlight else BASELINE for j in range(len(vals))]
        kw = dict(y=categories, x=vals, orientation="h") if horizontal else dict(x=categories, y=vals)
        if len(names) == 1 and len(categories) <= 3:
            kw["width"] = 0.3
        fig.add_trace(go.Bar(name=name, marker=dict(color=color),
                             text=text if len(names) == 1 else None, textposition="outside",
                             textfont=dict(color=INK_2, size=12), cliponaxis=False,
                             hovertemplate=(f"%{{{'x' if horizontal else 'y'}:{_HOVER.get(y_format, '')}}}"
                                            f"{_SUFFIX.get(y_format, '')}<extra>%{{{'y' if horizontal else 'x'}}}</extra>"),
                             **kw))
    fig.update_layout(barmode="group", bargap=0.45, bargroupgap=0.12)
    if threshold is not None:
        if horizontal:
            fig.add_vline(x=threshold, line=dict(color=MUTED, width=1))
        else:
            fig.add_hline(y=threshold, line=dict(color=MUTED, width=1))
            if threshold_label:
                fig.add_annotation(x=1, xref="paper", y=threshold, text=esc(threshold_label), showarrow=False,
                                   xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED))
    fig = _base(fig, height, title, y_format, legend=len(names) > 1)
    if y_format == "int":
        if horizontal:
            fig.update_xaxes(dtick=1)
        else:
            fig.update_yaxes(dtick=1)
    if horizontal and text:
        top = max((v for v in series[names[0]] if v is not None), default=1)
        fig.update_xaxes(range=[0, top * 1.18])  # room for the value labels
    if horizontal:
        fig.update_xaxes(tickformat=_TICK.get(y_format, ""), gridcolor=GRID, showgrid=True)
        fig.update_yaxes(showgrid=False, tickformat="", ticksuffix="", autorange="reversed", tickfont=dict(color=INK_2))
    return fig


def heatmap(days: list[str], features: list[str], values: list[list[float]], title: str | None = None,
            height: int = 300, zmax: float = 0.3) -> go.Figure:
    steps = len(SEQ_BLUE) - 1
    scale = [[i / steps, c] for i, c in enumerate(SEQ_BLUE)]
    fig = go.Figure(go.Heatmap(x=days, y=features, z=values, colorscale=scale, zmin=0, zmax=zmax, xgap=2, ygap=2,
                               colorbar=dict(title=dict(text="PSI", font=dict(size=11, color=MUTED)), thickness=10,
                                             outlinewidth=0, tickfont=dict(color=MUTED, size=11)),
                               hovertemplate="%{y} · %{x}<br><b>PSI %{z:.2f}</b><extra></extra>"))
    fig = _base(fig, height, title, "num", legend=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickformat="", rangemode="normal", tickfont=dict(color=INK_2))
    return fig


def chart_from_spec(spec: dict, height: int = 220) -> go.Figure | None:
    """Render a chart spec produced by the diagnosis engine."""
    if not spec:
        return None
    if spec["type"] == "line":
        return line_chart(spec["x"], {s["name"]: s["y"] for s in spec["series"]}, spec["y_format"], spec.get("title"),
                          height, markers=spec.get("markers"), threshold=spec.get("threshold"),
                          threshold_label=spec.get("threshold_label", ""))
    if spec["type"] == "bar":
        return bar_chart(spec["categories"], {s["name"]: s["values"] for s in spec["series"]}, spec["y_format"],
                         spec.get("title"), height, threshold=spec.get("threshold"),
                         threshold_label=spec.get("threshold_label", ""))
    return None


def plot(fig: go.Figure | None, key: str | None = None) -> None:
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
