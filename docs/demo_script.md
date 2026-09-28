# Two-minute demo: shot list and voice-over

**Story:** a client says their approval rate jumped. In under two minutes we go from alerts to a confirmed root cause, a plan with owners, and a customer update, and the system learns from it.

Target length is 1:55–2:05. LinkedIn autoplays muted, so burn in captions. The voice-over below is about 270 words, which runs roughly 2 minutes at a calm pace.

## Before you record

- [ ] Click **Reset demo data** in the sidebar (or run `python -m backend.app.seed`) so the data ends "now" and both demo incidents are untouched.
- [ ] Browser window about 1600×900, zoom 100%, sidebar open. Close other tabs and silence notifications.
- [ ] Do one dry run: `python scripts/ui_smoke_test.py --headed` walks the same path. Then reset again.
- [ ] Recorder: Windows Snipping Tool (Win+Shift+R → record), OBS, or Clipchamp. Record the voice separately if you can, since it is easier to re-take.

## Shot list

| Time | Screen | Do | Voice-over |
|---|---|---|---|
| 0:00–0:10 | **Model health** | Hold on the two red "Critical" cards | "ML models rarely fail loudly. They keep returning scores while something upstream is broken. This is a copilot for the hour after the alert." |
| 0:10–0:22 | **Model health**, credit charts | Point at the approval-rate band and the missing-value spike | "Our client, a lender, runs a credit model. Approval rate is up seven points, and one input started arriving empty on Saturday." |
| 0:22–0:38 | **Alerts & triage** | Tick the two credit alerts → *Use sample customer report* → *Open incident* | "I group the related alerts, add what the customer told us — West-region approvals up fifteen points — and open a tracked incident. The diagnosis runs immediately." |
| 0:38–1:05 | **Diagnosis** | Pause on the 94% card, scroll the evidence slowly | "Seven checks later: upstream schema change, 94%. Why? Every null comes from one data source, the new bureau API. It started fourteen minutes after the client's own change ticket. The connector logs name the renamed field. And incomplete applications are approved 79% of the time instead of 63%. Drift, deploys and serving are all ruled out, and it links a similar incident from April with the fix that worked." |
| 1:05–1:20 | **Action plan** | Click *Done* on the two P0 mitigations; status flips to Mitigated | "Every action has an owner — us or the client — and a reason tied to the evidence. Once the P0 mitigations are done, the incident moves to mitigated." |
| 1:20–1:40 | **Updates** | Show the customer update, toggle to engineering, back, *Mark as sent* | "Two updates from the same facts. For the client: what happened, what it means for them, what we need from them — no jargon. For engineers: evidence, alternatives, blast radius." |
| 1:40–1:52 | **Resolve & learn** → **Knowledge base** | Resolve, then show the accuracy tile and the new card | "I confirm the root cause. It goes into the knowledge base, and the next similar incident ranks faster." |
| 1:52–2:00 | README on GitHub, or the diagnosis view | Hold | "FastAPI, Streamlit, real scikit-learn models on synthetic data. Link in the comments." |

## Captions (short, for muted autoplay)

1. Models fail quietly. This is the hour after the alert.
2. Credit approvals +7pp · one input arriving empty
3. Alerts + customer report → tracked incident
4. Root cause: upstream schema change (94%)
5. Nulls from one source · 14 min after client change · logs confirm
6. Owners + reasons → P0s done → Mitigated
7. Customer update, no jargon · engineer update, full evidence
8. Resolved → knowledge base → next diagnosis is faster

## Screenshots to post (in this order)

1. `03-diagnosis.png`: the hero image (verdict and evidence)
2. `05-customer-update.png`: shows the FDE angle
3. `01-model-health.png`
4. `04-action-plan.png`

Re-generate them any time with `python scripts/capture_screenshots.py` (API and UI running).

## Launch checklist

- [ ] Run it from a clean clone: `pip install -r requirements.txt`, start the API and UI, click through once
- [ ] `docker compose up --build` works on a machine with Docker
- [ ] Push to GitHub; set the repo description and topics (`mlops`, `incident-response`, `fastapi`, `streamlit`)
- [ ] Record, caption and export the video (1080p, under 2:05)
- [ ] Post on LinkedIn (see `linkedin_post.md`) with the video natively uploaded and the repo link in the first comment
