# RoadPass — demo runbook & evidence

Use this checklist to prove **detect → server → ahead warning → (optional) longitudinal** before or during a hackathon demo. Order matters.

**Stack (this fork):**

- API base: `https://roadpass.jpadams.xyz`
- Report: `POST /events`, confirm: `PATCH /events/response`
- Ahead: `GET /hazards/ahead`
- Manual UI trigger (onroad, SSH): `touch /tmp/hazard_trigger`
- Longitudinal caps (optional): param **`RoadPassLongitudinalEnabled`** (bool). Requires a build that registers this key in `Params`; if the key is unknown, the feature stays off safely.

---

## Crowd scoring (what the UI shows)

The ahead-hazard card shows a **Score %**, a **tier** (High / Medium / Low), and **yes / responded** when counts exist.

| Source | Meaning |
|--------|---------|
| `response_summary.yes`, `no`, `timeout` | **Score** = percent of drivers who tapped **Yes** among those who answered (timeouts count as not yes). |
| `response_summary.yes` + `total` only | **Score** = `round(100 * yes / total)`; **total** is treated as the number of reports (legacy). |
| Optional `crowd_score` on each hazard | Server-supplied 0–100; shown as the headline score; tier still uses counts when present. |

Tier rules (UI reliability band): **High** if score ≥ 70% with at least 2 responses; **Medium** if score ≥ 40% or ≥ 3 responses; else **Low**. If there are no counts but `crowd_score` exists, tier is derived from score alone.

Implementation: `selfdrive/ui/onroad/hazard_scoring.py`.

---

## Comma 1 — detection quality (instrumentation)

Openpilot logs lines prefixed with **`roadpass.comma1`** (via `cloudlog` / swaglog). After each driver answer you also get a **running summary**: bump vs manual triggers, detect count, yes/no/timeout, POST/PATCH success vs fail, and **yes_rate** among drivers who responded.

**Quick file snapshot on device** (no log parsing):

```bash
cat /tmp/roadpass_comma1_metrics.json
```

Updated whenever a bump fires, a manual trigger runs, a report starts, a driver responds, or HTTP completes. Use it to tune `bump_detector.py` thresholds (`BUMP_ACCEL_THRESHOLD`, `BUMP_JERK_THRESHOLD`, `BUMP_COOLDOWN`) and to estimate false positives (high **no** + **timeout** vs **yes**).

Implementation: `selfdrive/ui/onroad/hazard_detection_metrics.py`.

---

## Prerequisites

- [ ] Device onroad (or full UI + stack running) with GPS fix for ahead fetches
- [ ] Network reachability to `roadpass.jpadams.xyz`
- [ ] SSH to the comma (or shell on device) for manual trigger
- [ ] Access to server logs / DB / admin UI for **Comma 1** evidence (or teammate confirms)

---

## Demo steps (tick as you go)

| Step | What to do | Pass |
|------|------------|------|
| **1 — Report (Comma 1)** | Onroad: `touch /tmp/hazard_trigger`. Tap **Yes** on the hazard popup (or document **No** / timeout if testing those paths). | Popup appears and dismisses correctly |
| **2 — Server** | On backend, confirm a new event: `POST /events` succeeded and, after **Yes**, `PATCH /events/response` (or your pipeline) shows `event_id` + confirmation. | Log row or API trace with `event_id` + timestamp |
| **3 — Ahead (Comma 2)** | Drive second session / other device / return after seed data: GPS fix on, same road corridor as server expects. | **“Hazard ahead · Xm”** card with **Score %** + tier (top-right) |
| **4 — Longitudinal (optional)** | Offroad: enable **`RoadPassLongitudinalEnabled`**. Repeat **3** with openpilot longitudinal active. | Softer accel or lower effective cruise vs same scene with param **off** |

---

## Evidence to send teammates / judges

Collect at least:

1. **One screenshot or screen recording** showing the confirmation popup (**step 1**).
2. **One server-side artifact**: log line, DB screenshot, or copied JSON with `event_id` and time (**step 2**).
3. **One screenshot or short clip** of the **Hazard ahead** card with distance **and crowd score line** (**step 3**).
4. *(Optional)* **Before/after** for **step 4** (param off vs on) — photo or cabana note.

Caption each with: date, branch/commit, and which step it proves.

---

## Commands reference

```bash
# Manual hazard popup (device must be onroad; UI will remove the file)
touch /tmp/hazard_trigger
```

```bash
# Enable longitudinal caps (offroad; method depends on your param tooling)
# Example if using comma’s params CLI / SSH helper — replace with what you use:
# params put Bool RoadPassLongitudinalEnabled true
```

```bash
# Watch UI / planner logs (adjust for your logging setup)
# tail -f /data/log/*  # example only
```

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| No popup after `touch` | Onroad? UI running? File path exactly `/tmp/hazard_trigger` |
| No server row | Network, TLS, `cloudlog` / reporter errors on device |
| No ahead card | GPS `hasFix`, hazard actually in API for this area, speed/warn-distance buckets |
| Longitudinal unchanged | Param on? OP longitudinal active? `plannerd` subscribed to latest `uiDebug`? Rebuild after adding param key |
| Metrics file missing | UI not run yet, or `/tmp` not writable — drive onroad once after boot |

---

## One-liner pitch (after demo)

> “Another comma user confirms a bump; we store it; your comma polls ahead hazards, shows a warning, and optionally softens longitudinal before the hazard.”
