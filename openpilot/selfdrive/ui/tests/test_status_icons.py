import pytest

from openpilot.cereal import log, messaging
from openpilot.selfdrive.ui.personality import personality_bar_count
from openpilot.selfdrive.ui.plan_source import SpeedLimiter, speed_limiter_from_source


LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource


@pytest.mark.parametrize(("source", "expected"), (
  (LongitudinalPlanSource.cruise, SpeedLimiter.CRUISE),
  (LongitudinalPlanSource.lead0, SpeedLimiter.LEAD),
  (LongitudinalPlanSource.lead1, SpeedLimiter.LEAD),
  (LongitudinalPlanSource.lead2, SpeedLimiter.LEAD),
  (LongitudinalPlanSource.e2e, SpeedLimiter.E2E),
))
def test_speed_limiter_from_source(source, expected):
  assert speed_limiter_from_source(source) == expected


@pytest.mark.parametrize(("personality", "expected"), (
  (log.LongitudinalPersonality.relaxed, 1),
  (log.LongitudinalPersonality.standard, 2),
  (log.LongitudinalPersonality.aggressive, 3),
))
def test_personality_bar_count(personality, expected):
  msg = messaging.new_message("selfdriveState")
  msg.selfdriveState.personality = personality

  assert personality_bar_count(msg.selfdriveState.personality) == expected


def test_personality_bar_count_unavailable():
  assert personality_bar_count(log.LongitudinalPersonality.aggressive, available=False) == 0


def test_personality_bar_count_unknown():
  assert personality_bar_count(None) == 0
