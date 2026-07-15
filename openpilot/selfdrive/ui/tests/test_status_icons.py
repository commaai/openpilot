import pytest

from openpilot.cereal import log
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
