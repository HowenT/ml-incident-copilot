"""Capture README / LinkedIn screenshots of the running app.

Prereqs: API on :8000 and UI on :8501 (see README), `pip install playwright`.
Uses an installed Edge or Chrome, so no browser download is needed:

    python scripts/capture_screenshots.py                 # all pages
    python scripts/capture_screenshots.py --only health   # one page
    python scripts/capture_screenshots.py --channel chrome

The script opens the credit incident through the API (if it is not open yet) so the
incident workspace has something to show, then walks each view.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

UI = "http://localhost:8501"
API = "http://localhost:8000"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def ensure_demo_incident() -> int:
    incs = httpx.get(f"{API}/api/incidents").json()
    for i in incs:
        if i["model_id"] == "credit-risk":
            return i["id"]
    alerts = [a for a in httpx.get(f"{API}/api/alerts?status=open&model_id=credit-risk").json()]
    sample = httpx.get(f"{API}/api/alerts/sample-report", params={"alert_ids": alerts[0]["id"]}).json()
    inc = httpx.post(f"{API}/api/incidents", json={**sample, "alert_ids": [a["id"] for a in alerts]}, timeout=60).json()
    httpx.post(f"{API}/api/incidents/{inc['id']}/diagnose", timeout=120)
    return inc["id"]


def settle(page: Page, text: str | None = None, extra: float = 2.5) -> None:
    if text:
        page.get_by_text(text, exact=False).first.wait_for(timeout=60_000)
    page.wait_for_load_state("networkidle")
    # Streamlit re-renders after websocket messages; give Plotly time to draw.
    deadline = time.time() + 20
    while time.time() < deadline and page.locator("[data-testid='stStatusWidget']").count():
        time.sleep(0.3)
    time.sleep(extra)


def shot(page: Page, name: str, height: int = 1800) -> None:
    page.set_viewport_size({"width": 1440, "height": height})
    # wait until the number of rendered Plotly charts stops changing
    last, stable = -1, 0
    for _ in range(40):
        n = page.locator(".js-plotly-plot .main-svg").count()
        stable = stable + 1 if n == last else 0
        if stable >= 4:
            break
        last = n
        time.sleep(0.5)
    page.mouse.move(2, 2)  # dismiss hover tooltips
    time.sleep(1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT / f"{name}.png"))
    print("saved", OUT / f"{name}.png")


def select_first_rows(page: Page, n: int) -> None:
    """Tick the row checkboxes of the first st.dataframe (a canvas grid, so click by position)."""
    box = page.locator("[data-testid='stDataFrame']").first.bounding_box()
    row_h = 35
    for i in range(n):
        page.mouse.click(box["x"] + 18, box["y"] + row_h * (i + 1) + row_h / 2)
        time.sleep(1.5)


def workspace_view(page: Page, label: str) -> None:
    page.get_by_role("radio", name=label, exact=True).first.click()
    settle(page, extra=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    ap.add_argument("--channel", default="msedge")
    args = ap.parse_args()
    ensure_demo_incident()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        if args.only in ("all", "health"):
            page.goto(UI)
            settle(page, "Model health")
            shot(page, "01-model-health", 1900)
        if args.only in ("all", "alerts"):
            page.goto(f"{UI}/alerts")
            settle(page, "Alerts & triage")
            select_first_rows(page, 2)
            settle(page, "Open an incident")
            page.get_by_role("button", name="Use sample customer report").click()
            settle(page, extra=2)
            shot(page, "02-alerts-triage", 1750)
        if args.only in ("all", "workspace"):
            page.goto(f"{UI}/incident")
            settle(page, "Most likely root cause")
            workspace_view(page, "Diagnosis")
            shot(page, "03-diagnosis", 2300)
            workspace_view(page, "Action plan")
            shot(page, "04-action-plan", 1900)
            workspace_view(page, "Updates")
            shot(page, "05-customer-update", 1500)
            page.get_by_role("radio", name="Engineering diagnosis (technical)").click()
            settle(page, extra=2)
            shot(page, "06-engineer-summary", 1900)
            workspace_view(page, "Overview")
            shot(page, "07-overview-timeline", 1700)
        if args.only in ("all", "knowledge"):
            page.goto(f"{UI}/knowledge")
            settle(page, "Knowledge base")
            shot(page, "08-knowledge-base", 1500)
        browser.close()


if __name__ == "__main__":
    main()
