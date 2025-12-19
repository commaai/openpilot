import pytest

from openpilot.tools.bodyteleop.gemini_plan import (
  parse_plan_from_text,
  validate_plan,
  plan_end_time_seconds,
  compute_next_gemini_call_time,
)


def test_parse_plan_happy_path():
  txt = """plan
1,0,0,0,1.0
0,0,0,1,1.3
0,0,0,0,1.4
"""
  plan = parse_plan_from_text(txt)
  assert plan is not None
  assert plan == [(1, 0, 0, 0, 1.0), (0, 0, 0, 1, 1.3), (0, 0, 0, 0, 1.4)]


def test_parse_plan_rejects_multiple_directions_per_row():
  txt = """plan
1,0,0,1,0.5
"""
  assert parse_plan_from_text(txt) is None


def test_parse_plan_rejects_non_monotonic_timestamps():
  txt = """plan
1,0,0,0,1.0
0,1,0,0,0.9
"""
  assert parse_plan_from_text(txt) is None


def test_parse_plan_rejects_over_5s():
  txt = """plan
1,0,0,0,5.1
"""
  assert parse_plan_from_text(txt) is None


def test_validate_plan_basic():
  assert validate_plan([(0, 0, 0, 0, 0.1)]) is True
  assert validate_plan([]) is False


def test_plan_end_time_seconds():
  assert plan_end_time_seconds([(0, 0, 0, 0, 0.1)]) == 0.1
  assert plan_end_time_seconds([(1, 0, 0, 0, 1.0), (0, 0, 0, 0, 1.2)]) == 1.2
  assert plan_end_time_seconds(None) == 0.0


def test_compute_next_gemini_call_time_min_interval_only():
  last_call = 100.0
  assert compute_next_gemini_call_time(last_call, 10.0, None, None) == 110.0


def test_compute_next_gemini_call_time_waits_for_plan_end():
  last_call = 100.0
  plan_start = 103.0
  plan = [(1, 0, 0, 0, 1.0), (0, 0, 0, 0, 2.5)]
  # min interval => 110.0, plan end => 105.5, should pick 110.0
  assert compute_next_gemini_call_time(last_call, 10.0, plan_start, plan) == 110.0

  # if min interval is smaller, should wait for plan end
  assert compute_next_gemini_call_time(last_call, 1.0, plan_start, plan) == pytest.approx(105.5)


