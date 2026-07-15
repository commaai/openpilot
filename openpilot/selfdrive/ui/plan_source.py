from enum import Enum

from openpilot.cereal import log


LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource


class SpeedLimiter(Enum):
  CRUISE = "cruise"
  LEAD = "lead"
  E2E = "e2e"


def speed_limiter_from_source(source) -> SpeedLimiter:
  if source in (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1, LongitudinalPlanSource.lead2):
    return SpeedLimiter.LEAD
  if source == LongitudinalPlanSource.e2e:
    return SpeedLimiter.E2E
  return SpeedLimiter.CRUISE
