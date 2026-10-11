import time

import numpy as np
import pyray as rl

from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import MapRenderer, Position
from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.navigation.route import RouteTracker, route_coordinates
from openpilot.selfdrive.navigation.tiles import RemoteTiles
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.text_measure import measure_text_cached

PANEL_WIDTH = 261
STALE_SECONDS = 3


class NavigationPanel(Widget):
  def __init__(self):
    super().__init__()
    self._archive = None
    self._renderer = None
    self._route = np.empty((0, 2))
    self._tracker: RouteTracker | None = None
    self._instruction = {}
    self._next_render = 0.0
    self._frame_timestamp = 0
    self._texture = None
    self._map_size: tuple[int, int] | None = None  # on-screen map pixels, known after the first render
    self._font = gui_app.font(FontWeight.MEDIUM)
    self._bold = gui_app.font(FontWeight.BOLD)
    self._accent = rl.Color(155, 230, 185, 255)
    self._muted = rl.Color(162, 175, 185, 255)

  def update(self):
    sm = ui_state.sm
    if sm.updated['navInstruction']:
      self._instruction = sm['navInstruction'].to_dict()
    if sm.updated['navRoute']:
      route = route_coordinates(sm['navRoute']) if sm.valid['navRoute'] else np.empty((0, 2))
      if not np.array_equal(route, self._route):
        self._route = route
        self._tracker = RouteTracker(route) if len(route) >= 2 else None
        self._frame_timestamp = 0
        self._next_render = 0.0
    now = time.monotonic()
    if now < self._next_render or self._map_size is None or not self.active:
      return
    self._next_render = now + 0.2
    source = latest_gps_source(sm)
    if source is None:
      self._frame_timestamp = 0
      return
    gps = sm[source]
    try:
      if self._renderer is None:
        self._archive = PMTiles(RemoteTiles(online=False))  # mapsd fetches the route's tiles into the shared cache
        self._renderer = MapRenderer(self._archive)
      if self._tracker is not None:
        # The same route-snapped pose the model sees.
        longitude, latitude, bearing = self._tracker.snap(gps.longitude, gps.latitude)
        position = Position(latitude, longitude, bearing)
      else:
        position = Position(gps.latitude, gps.longitude, gps.bearingDeg)
      width, height = self._map_size
      result = self._renderer.render(position, self._route, width=width, height=height, route_color=(72, 235, 132), vehicle=False)
      if result.valid:
        buffer = rl.ffi.new('unsigned char[]', result.image.tobytes())
        texture = rl.load_texture_from_image(rl.Image(buffer, width, height, 1, rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8))
        if self._texture:
          rl.unload_texture(self._texture)
        self._texture = texture
        self._frame_timestamp = int(sm.recv_time[source] * 1e9)
    except (OSError, ValueError):
      # Not cached yet, or a new archive version: try again once mapsd has fetched it.
      if self._archive is not None:
        self._archive.close()
      self._archive = self._renderer = None
      self._frame_timestamp = 0
      self._next_render = now + 2

  @property
  def active(self):
    return ui_state.navigation_enabled and ui_state.sm.valid['navInstruction'] and ui_state.sm.alive['navInstruction']

  def _text(self, text, x, y, size, width, bold=False, color=None):
    font = self._bold if bold else self._font
    text = str(text)
    if measure_text_cached(font, text, size).x > width:
      while text and measure_text_cached(font, text + '...', size).x > width:
        text = text[:-1]
      text += '...'
    rl.draw_text_ex(font, text, rl.Vector2(x, y), size, 0, color or rl.WHITE)

  def _render(self, rect):
    x, y = rect.x, rect.y
    rl.draw_rectangle_rec(rect, rl.Color(7, 13, 18, 255))
    direction = str(self._instruction.get('maneuverModifier', 'straight'))
    flip = -1 if 'left' in direction else 1
    if direction in ('straight', ''):
      points = [(x+22,y+42), (x+22,y+16), (x+16,y+23), (x+22,y+16), (x+28,y+23)]
    else:
      points = [(x+22,y+42), (x+22,y+24), (x+22+flip*7,y+19), (x+22+flip*18,y+19),
                (x+22+flip*12,y+13), (x+22+flip*18,y+19), (x+22+flip*12,y+25)]
    for a, b in zip(points[:-1], points[1:], strict=True):
      rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), 3, self._accent)
    distance = max(0, self._instruction.get('maneuverDistance', 0))
    if ui_state.is_metric:
      distance_text = f'{round(distance / 10) * 10} m' if distance < 1000 else f'{distance / 1000:.1f} km'
    else:
      distance_text = f'{round(distance * 3.28084 / 50) * 50} ft' if distance < 300 else f'{distance / 1609.344:.1f} mi'
    self._text(distance_text, x+46, y+8, 19, rect.width-58, bold=True)
    self._text(self._instruction.get('maneuverPrimaryText', ''), x+46, y+34, 17, rect.width-58, bold=True)
    map_rect = rl.Rectangle(x+8, y+61, rect.width-16, rect.height-95)
    rl.draw_rectangle_rec(map_rect, rl.BLACK)
    # The renderer places ego at the image's bottom center. Leave room below
    # that origin so the whole marker is visible without shifting it off-route.
    image_rect = rl.Rectangle(map_rect.x, map_rect.y, map_rect.width, map_rect.height-18)
    # Render at the on-screen size, so the texture is drawn 1:1 without resampling.
    self._map_size = (int(image_rect.width), int(image_rect.height))
    age = time.monotonic() - self._frame_timestamp / 1e9
    if self._texture and 0 <= age < STALE_SECONDS:
      rl.draw_texture_pro(self._texture, rl.Rectangle(0, 0, self._texture.width, self._texture.height), image_rect, rl.Vector2(0, 0), 0, rl.WHITE)
      cx, cy = image_rect.x + image_rect.width/2, image_rect.y + image_rect.height
      rl.draw_triangle(rl.Vector2(cx, cy-14), rl.Vector2(cx-11, cy+10), rl.Vector2(cx+11, cy+10), rl.BLACK)
      rl.draw_triangle(rl.Vector2(cx, cy-10), rl.Vector2(cx-7, cy+7), rl.Vector2(cx+7, cy+7), rl.Color(255, 210, 64, 255))
    else:
      self._text('Map unavailable', x+16, y+125, 11, 142, color=self._muted)
    seconds = max(0, self._instruction.get('timeRemaining', 0))
    remaining = max(0, self._instruction.get('distanceRemaining', 0))
    length = f'{remaining / 1000:.1f} km' if ui_state.is_metric else f'{remaining / 1609.344:.1f} mi'
    self._text(f'{max(1, round(seconds / 60))} min / {length}', x+9, y+rect.height-28, 19, rect.width-18, bold=True)

  def close(self):
    if self._texture:
      rl.unload_texture(self._texture)
      self._texture = None
    if self._archive:
      self._archive.close()
      self._archive = None
      self._renderer = None
