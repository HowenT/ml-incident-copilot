# LinkedIn post drafts

Upload the video natively and put the GitHub link in the first comment, since LinkedIn down-ranks posts with external links. Replace `[link]` before posting.

---

## Version A: story (recommended)

ML models rarely fail with an error message.

They fail like this: a client's IT team switches West-region credit checks to a new bureau API. The new API renames one field. The connector sends nulls, the pipeline quietly fills in an average value, and risky applicants start looking average. No alarms. The approval rate is simply up 15 points in one region, and the risk team wants to know why.

The dashboard isn't the hard part. The hour after the alert is: which alerts belong together, what actually broke, who has to do what, and what you tell the customer.

So I built an ML Incident Copilot that runs that hour as a workflow:

🔎 Triage related alerts into one tracked incident, with the customer's own description of the impact
🧪 Explainable diagnosis across 7 layers: data quality, drift, decisions, labels, serving, change events, logs. It ranks root causes with the evidence for and against each
✅ An action plan with owners (us or the client) and a reason behind every step
✉️ Two updates from the same facts: technical for engineers, plain language for the business owner
🔁 A feedback loop: confirmed root causes improve the next diagnosis

On three realistic incidents (a vendor schema change, a slow model release, and a new card programme the model had never seen), it ranks the right root cause first and pins sudden failures to the hour they started.

Three things I learned building it:
1. Most root causes come from localisation, not from the metric itself. Nulls from one data source, false declines from one card programme, latency from one model version.
2. Label lag is real. Credit outcomes arrive weeks later, so a good diagnosis says so and reasons from other signals.
3. Use an LLM for the words, not the verdict. The diagnosis is rule-based and tested. Claude optionally polishes the customer update, with every number pinned.

Stack: FastAPI · scikit-learn · Streamlit · SQLite · Docker. The data is synthetic and the models are real.

Code and 2-min demo in the comments. I'd love feedback from anyone who runs ML in production. What would you add?

#MLOps #MachineLearning #AIEngineering #ForwardDeployedEngineering #IncidentResponse

---

## Version B: short

The hardest part of ML in production isn't building the dashboard. It's the hour after the alert.

I built an ML Incident Copilot for that hour:
→ alerts + the customer's report become one tracked incident
→ an explainable diagnosis across 7 layers (data, drift, decisions, labels, serving, changes, logs) ranks root causes with evidence for and against
→ an action plan with owners and reasons
→ one update for engineers and one in plain language for the client
→ resolved incidents make the next diagnosis better

Demo: a client's vendor API renamed one field, and approvals quietly jumped 15 points. Root cause, blast radius and a customer update in under two minutes.

FastAPI · scikit-learn · Streamlit · Docker. Repo in the comments.

#MLOps #AIEngineering #ForwardDeployedEngineering

---

## First comment

Code: [link] · 2-min walkthrough in the post. Everything runs locally in about a minute (`docker compose up`). The data and client are fictional; the models are real scikit-learn models trained on simulated populations.
