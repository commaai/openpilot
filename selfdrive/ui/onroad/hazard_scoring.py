"""
Crowd scoring for RoadPass hazards (client-side, from API `response_summary`).

The server may also send a precomputed `crowd_score` (0–100) on each hazard object;
if present it is shown as the headline score; tier uses response counts when available.

Formulas (explicit for demos and server alignment):

1. **responded** = yes + no + timeout (drivers who completed the popup).

2. **score_pct** (confirmation rate):
   - If `crowd_score` is on the hazard JSON: use it, clamped to 0–100.
   - Else if there are **no** `no` or `timeout` counts but **`total` > 0** (legacy `{yes, total}`):
     `round(100 * yes / total)` — **total** is the denominator (e.g. 3 yes / 4 asked).
   - Else if **responded** > 0: `round(100 * yes / responded)`. Timeouts count as not confirming.

3. **Tier** (reliability band for the UI):
   - **high**: score_pct ≥ 70 and responded ≥ 2 (or score-only path: score_pct ≥ 70)
   - **med**:  score_pct ≥ 40 or responded ≥ 3 (or score-only med band)
   - **low**:  otherwise

If there is no usable data, `hazard_score_from_api` returns None.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HazardScore:
  score_pct: int
  tier: str  # "high" | "med" | "low"
  yes: int
  no: int
  timeout: int
  responded: int

  @property
  def tier_label(self) -> str:
    return {"high": "High", "med": "Medium", "low": "Low"}.get(self.tier, self.tier)


def _tier_from_score_and_responded(score_pct: int, responded: int) -> str:
  if responded <= 0:
    return "high" if score_pct >= 70 else "med" if score_pct >= 40 else "low"
  if score_pct >= 70 and responded >= 2:
    return "high"
  if score_pct >= 40 or responded >= 3:
    return "med"
  return "low"


def hazard_score_from_api(hazard_dict: dict) -> HazardScore | None:
  """
  Build a HazardScore from one element of `GET /hazards/ahead` JSON `hazards[]`.
  """
  summary = hazard_dict.get("response_summary") or {}
  yes = int(summary.get("yes", 0) or 0)
  no = int(summary.get("no", 0) or 0)
  timeout = int(summary.get("timeout", 0) or 0)
  responded = yes + no + timeout

  raw_cs = hazard_dict.get("crowd_score")
  if raw_cs is not None:
    try:
      score_pct = int(round(float(raw_cs)))
    except (TypeError, ValueError):
      score_pct = 0
    score_pct = max(0, min(100, score_pct))
    tier = _tier_from_score_and_responded(score_pct, responded)
    return HazardScore(
      score_pct=score_pct, tier=tier, yes=yes, no=no, timeout=timeout, responded=responded,
    )

  total = int(summary.get("total", 0) or 0)
  # Legacy payloads: only yes + total (no per-outcome no/timeout breakdown).
  if no == 0 and timeout == 0 and total > 0:
    score_pct = int(round(100.0 * yes / max(1, total)))
    score_pct = max(0, min(100, score_pct))
    tier = _tier_from_score_and_responded(score_pct, total)
    return HazardScore(
      score_pct=score_pct, tier=tier, yes=yes, no=no, timeout=timeout, responded=total,
    )

  if responded <= 0:
    return None

  score_pct = int(round(100.0 * yes / responded))
  score_pct = max(0, min(100, score_pct))
  tier = _tier_from_score_and_responded(score_pct, responded)
  return HazardScore(
    score_pct=score_pct, tier=tier, yes=yes, no=no, timeout=timeout, responded=responded,
  )
