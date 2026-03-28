import time
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.onroad.hazard_fetcher import HazardFetcher, HazardAhead, _bearing_delta, _speed_to_bucket
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Small floating card — does not cover the road view or HUD.
CARD_WIDTH = 560
CARD_HEIGHT = 118
CARD_MARGIN = 30        # from the right and top edges of the content rect
CARD_RADIUS = 16        # corner rounding in pixels
FONT_SIZE_MAIN = 46
FONT_SIZE_SUB = 34

CARD_BG = rl.Color(170, 85, 0, 230)    # dark amber, semi-transparent

# How long to display the warning for a single hazard before suppressing it.
SHOW_TIMEOUT = 15.0  # seconds

# Bearing window that counts as "ahead".
AHEAD_THRESHOLD_DEG = 90.0


class HazardAheadRenderer(Widget):
  """
  Renders a small floating notification card in the top-right corner of the
  onroad view when a known hazard is ahead and within the speed-appropriate
  warning distance.

  Per-hazard lifecycle:
    - First appears when the hazard enters warning range and is ahead.
    - Suppressed (removed) when any of the following occur:
        1. The hazard passes behind the device (bearing delta > 90°).
        2. The warning has been visible for 15 seconds (driver has been warned).
    - Suppression persists for the remainder of the drive — the hazard will
      not reappear even if it is returned by a subsequent API fetch.
  """

  def __init__(self, fetcher: HazardFetcher):
    super().__init__()
    self._fetcher = fetcher
    # event_id → monotonic time when the card first became visible
    self._show_start: dict[str, float] = {}
    # event_ids that have been dismissed and should never show again
    self._suppressed: set[str] = set()

  def _render(self, rect: rl.Rectangle):
    gps = ui_state.sm['gpsLocationExternal']
    if not gps.hasFix:
      return

    hazards = self._fetcher.get_hazards()
    if not hazards:
      return

    device_lat = gps.latitude
    device_lon = gps.longitude
    device_bearing = gps.bearingDeg
    speed_ms = ui_state.sm['carState'].vEgo

    _, warn_distance_m = _speed_to_bucket(speed_ms)

    closest = self._find_closest(hazards, device_lat, device_lon, device_bearing, warn_distance_m)
    if closest is None:
      return

    hazard, dist = closest
    self._draw_card(rect, hazard, dist)

  def _find_closest(
    self,
    hazards: list[HazardAhead],
    device_lat: float,
    device_lon: float,
    device_bearing: float,
    warn_distance_m: int,
  ) -> tuple[HazardAhead, float] | None:
    now = time.monotonic()
    closest_hazard = None
    closest_dist = float('inf')

    for hazard in hazards:
      if hazard.event_id in self._suppressed:
        continue

      dist = hazard.distance_m(device_lat, device_lon)
      bearing_to = hazard.bearing_deg(device_lat, device_lon)

      # Suppress if the hazard is now behind us.
      if _bearing_delta(device_bearing, bearing_to) > AHEAD_THRESHOLD_DEG:
        self._suppressed.add(hazard.event_id)
        continue

      if dist > warn_distance_m:
        continue

      # Start the visibility clock the first time this hazard enters range.
      if hazard.event_id not in self._show_start:
        self._show_start[hazard.event_id] = now

      # Suppress after the timeout.
      if now - self._show_start[hazard.event_id] > SHOW_TIMEOUT:
        self._suppressed.add(hazard.event_id)
        continue

      if dist < closest_dist:
        closest_hazard = hazard
        closest_dist = dist

    return (closest_hazard, closest_dist) if closest_hazard is not None else None

  def get_active_ahead_hazard(
    self,
    device_lat: float,
    device_lon: float,
    device_bearing: float,
    speed_ms: float,
  ) -> tuple[HazardAhead, float, int] | None:
    """
    Same hazard and distance as the on-screen warning card (including suppression rules).
    Returns (hazard, distanceM, warnDistanceM) or None.
    """
    hazards = self._fetcher.get_hazards()
    if not hazards:
      return None
    _, warn_distance_m = _speed_to_bucket(speed_ms)
    pair = self._find_closest(hazards, device_lat, device_lon, device_bearing, warn_distance_m)
    if pair is None:
      return None
    hazard, dist = pair
    return hazard, dist, warn_distance_m

  def _draw_card(self, rect: rl.Rectangle, hazard: HazardAhead, distance_m: float) -> None:
    # Position in the top-right corner of the content rect.
    x = rect.x + rect.width - CARD_WIDTH - CARD_MARGIN
    y = rect.y + CARD_MARGIN
    card_rect = rl.Rectangle(x, y, CARD_WIDTH, CARD_HEIGHT)

    roundness = CARD_RADIUS / (min(CARD_WIDTH, CARD_HEIGHT) / 2)
    rl.draw_rectangle_rounded(card_rect, roundness, 10, CARD_BG)

    dist_str = f"{int(distance_m)}m"
    line_main = f"Hazard ahead  \xb7  {dist_str}"
    if hazard.score is not None:
      sc = hazard.score
      if sc.responded > 0:
        line_sub = f"Score {sc.score_pct}% · {sc.tier_label} · {sc.yes}/{sc.responded} yes"
      else:
        line_sub = f"Score {sc.score_pct}% · {sc.tier_label} · server"
    else:
      line_sub = "Score: not enough driver votes yet"

    font = gui_app.font(FontWeight.BOLD)
    sz_m = measure_text_cached(font, line_main, FONT_SIZE_MAIN)
    sz_s = measure_text_cached(font, line_sub, FONT_SIZE_SUB)
    gap = 6
    block_h = sz_m.y + gap + sz_s.y
    ty0 = y + (CARD_HEIGHT - block_h) / 2
    tx_m = x + (CARD_WIDTH - sz_m.x) / 2
    tx_s = x + (CARD_WIDTH - sz_s.x) / 2
    rl.draw_text_ex(font, line_main, rl.Vector2(int(tx_m), int(ty0)), FONT_SIZE_MAIN, 0, rl.WHITE)
    rl.draw_text_ex(font, line_sub, rl.Vector2(int(tx_s), int(ty0 + sz_m.y + gap)), FONT_SIZE_SUB, 0, rl.WHITE)
