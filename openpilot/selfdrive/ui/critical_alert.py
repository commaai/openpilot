"""Sound-independent policy for sustained visible critical alerts.

Call on every soundd tick, including clear/silent states. Eligibility comes
from severity and visibility, never from the assigned sound or silence.
Changing the starting sound or alert identity does not restart the timer.
"""
from dataclasses import dataclass
from enum import IntEnum

CRITICAL_ESCALATION_TIME = 8


class MaxAlert(IntEnum):
  # Local playback IDs; never serialized as SelfdriveState.AudibleAlert.
  driver = -1
  critical = -2


def max_alert_for_type(alert_type: str) -> MaxAlert:
  """Route by event identity; unknown/new events always get the generic max."""
  event = alert_type.partition('/')[0]
  if event.startswith(('driverDistracted', 'driverUnresponsive', 'driverMonitoringPreview')):
    return MaxAlert.driver
  return MaxAlert.critical


@dataclass
class CriticalAlertEscalation:
  started_at: float | None = None

  def update(self, now: float, eligible: bool, max_alert: MaxAlert) -> MaxAlert | None:
    if not eligible:
      self.started_at = None
      return None
    if self.started_at is None:
      self.started_at = now
    if now - self.started_at >= CRITICAL_ESCALATION_TIME:
      return max_alert
    return None
