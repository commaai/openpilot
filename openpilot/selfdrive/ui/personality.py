from openpilot.cereal import log


def personality_bar_count(personality: log.LongitudinalPersonality, available: bool = True) -> int:
  """Map personality to aggressiveness intensity, not following-distance bars."""
  if not available:
    return 0
  if personality == log.LongitudinalPersonality.aggressive:
    return 3
  if personality == log.LongitudinalPersonality.standard:
    return 2
  if personality == log.LongitudinalPersonality.relaxed:
    return 1
  return 0
