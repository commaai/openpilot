from openpilot.selfdrive.ui.onroad.hazard_scoring import hazard_score_from_api


def test_legacy_yes_and_total_only():
  d = {
    "response_summary": {"yes": 3, "total": 4},
  }
  s = hazard_score_from_api(d)
  assert s is not None
  assert s.score_pct == 75
  assert s.responded == 4
  assert s.tier in ("high", "med", "low")


def test_responded_yes_no_timeout():
  d = {
    "response_summary": {"yes": 2, "no": 1, "timeout": 1},
  }
  s = hazard_score_from_api(d)
  assert s is not None
  assert s.score_pct == 50
  assert s.responded == 4


def test_crowd_score_override():
  d = {
    "response_summary": {"yes": 1, "total": 100},
    "crowd_score": 88,
  }
  s = hazard_score_from_api(d)
  assert s is not None
  assert s.score_pct == 88


def test_empty_returns_none():
  assert hazard_score_from_api({}) is None


def test_tier_high_when_unanimous():
  d = {"response_summary": {"yes": 3, "no": 0, "timeout": 0}}
  s = hazard_score_from_api(d)
  assert s is not None
  assert s.score_pct == 100
  assert s.tier == "high"
  assert s.tier_label == "High"
