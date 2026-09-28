"""Drive the full demo flow through the real UI (a rehearsal of the 2-minute video).

    python scripts/ui_smoke_test.py            # headless
    python scripts/ui_smoke_test.py --headed   # watch it

Needs the API (:8000) and UI (:8501) running on freshly seeded data and `pip install playwright`
(uses an installed Edge/Chrome, no browser download). Resets the demo data first.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx
from playwright.sync_api import Page, expect, sync_playwright

UI = "http://localhost:8501"
API = "http://localhost:8000"


def step(msg: str) -> None:
    print(f"• {msg}", flush=True)


def wait_idle(page: Page, extra: float = 1.5) -> None:
    page.wait_for_load_state("networkidle")
    time.sleep(extra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--channel", default="msedge")
    args = ap.parse_args()

    step("resetting demo data")
    httpx.post(f"{API}/api/admin/reseed", timeout=180).raise_for_status()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=not args.headed, slow_mo=150 if args.headed else 0)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(60_000)

        step("dashboard shows both models as critical")
        page.goto(UI)
        expect(page.get_by_text("Model health").first).to_be_visible()
        expect(page.get_by_text("Card transaction fraud").first).to_be_visible()

        step("triage: select the two latency alerts and use the sample report")
        page.get_by_role("link", name="Alerts & triage").click()
        expect(page.get_by_text("Select one or more alerts")).to_be_visible()
        wait_idle(page)
        box = page.locator("[data-testid='stDataFrame']").first.bounding_box()
        for i in range(2):
            page.mouse.click(box["x"] + 18, box["y"] + 35 * (i + 1) + 17)
            time.sleep(1.2)
        expect(page.get_by_text("2 alert(s) on Card transaction fraud")).to_be_visible()
        page.get_by_role("button", name="Use sample customer report").click()
        wait_idle(page)
        expect(page.get_by_label("Title")).to_have_value("Checkout latency and unscored approvals on card transactions")

        step("open the incident; the diagnosis runs and the workspace opens")
        page.get_by_role("button", name="Open incident").click()
        expect(page.get_by_text("Most likely root cause")).to_be_visible(timeout=90_000)
        expect(page.get_by_text("Model release regression").first).to_be_visible()
        expect(page.get_by_text("Degradation is confined to version 7.3.0").first).to_be_visible()

        step("action plan: complete both P0 mitigations -> status becomes Mitigated")
        page.get_by_role("radio", name="Action plan", exact=True).click()
        wait_idle(page)
        for idx in (1, 2):  # P0 order: verify, mitigate, mitigate, communicate
            page.get_by_role("radiogroup", name="Status").nth(idx).get_by_role("radio", name="Done").click()
            wait_idle(page, 2.5)
        expect(page.locator(".pill", has_text="Mitigated").first).to_be_visible()

        step("updates: customer update is plain language; mark it as sent")
        page.get_by_role("radio", name="Updates", exact=True).click()
        wait_idle(page)
        expect(page.get_by_text("What we need from you").first).to_be_visible()
        page.get_by_role("button", name="Mark as sent").click()
        wait_idle(page, 2.5)
        expect(page.get_by_text("· sent").first).to_be_visible()

        step("resolve & learn: confirm the root cause")
        page.get_by_role("radio", name="Resolve & learn", exact=True).click()
        wait_idle(page)
        page.get_by_role("button", name="Resolve incident").click()
        expect(page.get_by_text("Resolved — added to the knowledge base").first).to_be_visible()
        expect(page.get_by_text("Diagnosis was correct").first).to_be_visible()

        step("knowledge base now has 9 resolved incidents")
        page.get_by_role("link", name="Knowledge base").click()
        expect(page.get_by_text("Checkout latency and unscored approvals").first).to_be_visible()
        stats = httpx.get(f"{API}/api/knowledge/stats").json()
        assert stats["resolved"] == 9, stats
        browser.close()
    print("✓ UI flow passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
