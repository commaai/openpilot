"""Deterministic CPU rasterization of Protomaps roads and a supplied route.

Styled after the training map renderer: 1 m per pixel, the ego three quarters
down the frame, pure white roads whose casings are part of the stroke, bridges cutting over the roads beneath
them, a grey route and a white ego arrow.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.route import MAX_LATITUDE
from openpilot.selfdrive.navigation.vector_tile import Road, decode_roads

SIZE = 256
ZOOM = 15  # tile zoom; the style zoom is derived from METERS_PER_PIXEL
METERS_PER_PIXEL = 1.0
FORWARD_OFFSET = 64.0  # the ego sits three quarters down the frame
EARTH_CIRCUMFERENCE = 40075016.68557849
BACKGROUND = 0
ROAD_INTENSITY = 255
ROUTE_INTENSITY = 128
ROUTE_WIDTH = 5.0  # pixels
CUT_WIDTH = 3.0  # black separator under a bridge, added to the road width
MAX_ROAD_DISTANCE = 100.0  # meters from the ego to the nearest road centerline for a valid map
ARROW = np.array([(0, -9), (7, 7), (0, 3), (-7, 7)], dtype=np.float64)  # ego marker, pixels around the ego
ARROW_OUTLINE = 2.0  # pixels


def interpolate_width(zoom: float, *stops: float) -> float:
  """The style's exponential (base 1.5) interpolation between (zoom, width) stops, in screen pixels."""
  zooms, widths = stops[::2], stops[1::2]
  if zoom <= zooms[0]:
    return widths[0]
  if zoom >= zooms[-1]:
    return widths[-1]
  for low_zoom, low, high_zoom, high in zip(zooms, widths, zooms[1:], widths[1:], strict=False):
    if low_zoom <= zoom <= high_zoom:
      fraction = (1.5 ** (zoom - low_zoom) - 1) / (1.5 ** (high_zoom - low_zoom) - 1)
      return low + fraction * (high - low)
  raise ValueError("Unsorted width stops")


@dataclass(frozen=True)
class Style:
  interior: tuple[float, ...]  # (zoom, width, ...) stops
  casing: tuple[float, ...]  # one side, in pixels


# The training style's road classes. Casings are white there too, so they only widen the stroke.
MAJOR = Style((5, 0.75, 18, 32), (10, 1, 18, 2))  # motorway, trunk, primary
MEDIUM = Style((5, 0.1, 18, 26), (10, 0.75, 18, 2))  # secondary, tertiary
STREET = Style((12, 0.5, 14, 2, 18, 18), (12, 0.75, 20, 2))  # residential, unclassified, living streets, major links
MINOR = Style((14, 0.5, 18, 12), (12, 0.75, 20, 2))  # service roads and minor links
TRACK = Style((14, 1, 18, 12), (12, 0.75, 20, 2))
STYLES = (MAJOR, MEDIUM, STREET, MINOR, TRACK)


def road_style(road: Road) -> Style | None:
  """The style for a road, or None when hidden. Protomaps kind_detail is the underlying OSM highway type."""
  detail = road.detail
  if detail in ('platform', 'pier', 'raceway'):
    return None
  if detail in ('motorway', 'trunk', 'primary') or (not detail and road.kind == 'highway'):
    return MAJOR
  if detail in ('secondary', 'tertiary') or (not detail and road.kind == 'major_road'):
    return MEDIUM
  if detail == 'track':
    return TRACK
  if detail in ('secondary_link', 'tertiary_link', 'service', 'driveway', 'parking_aisle', 'alley') or (
    road.kind == 'other' and detail != 'living_street'):
    return MINOR
  return STREET


def road_level(road: Road) -> int:
  """Draw order: tunnels, then ground roads, then bridges, each cutting over the levels beneath."""
  return 1 if road.bridge else -1 if road.tunnel else 0


@dataclass(frozen=True)
class Position:
  latitude: float
  longitude: float
  bearing: float  # degrees clockwise from north

  def __post_init__(self):
    if not all(math.isfinite(v) for v in (self.latitude, self.longitude, self.bearing)):
      raise ValueError("Non-finite position")
    if abs(self.latitude) > MAX_LATITUDE or abs(self.longitude) > 180:
      raise ValueError("Position outside Web Mercator bounds")


@dataclass
class RenderResult:
  image: np.ndarray
  complete: bool  # every tile covering the view was present
  route_present: bool = False
  road_distance: float = math.inf  # meters from the ego to the nearest drawn road centerline

  @property
  def valid(self) -> bool:
    return self.complete and self.route_present and self.road_distance <= MAX_ROAD_DISTANCE


def mercator(coordinates: np.ndarray) -> np.ndarray:
  """Longitude/latitude -> normalized Web Mercator, x east / y south."""
  longitude, latitude = coordinates.T
  y = (1.0 - np.arcsinh(np.tan(np.deg2rad(latitude))) / np.pi) / 2.0
  return np.column_stack(((longitude + 180.0) / 360.0, y))


def project(points: np.ndarray, position: Position) -> np.ndarray:
  origin = mercator(np.array([[position.longitude, position.latitude]]))[0]
  delta = points - origin
  delta[:, 0] = (delta[:, 0] + 0.5) % 1.0 - 0.5  # wrap across the antimeridian
  delta *= EARTH_CIRCUMFERENCE * math.cos(math.radians(position.latitude))
  east, north = delta[:, 0], -delta[:, 1]
  angle = math.radians(position.bearing)
  right = east * math.cos(angle) - north * math.sin(angle)
  forward = east * math.sin(angle) + north * math.cos(angle)
  return np.column_stack((SIZE / 2 + right / METERS_PER_PIXEL,
                          SIZE / 2 - (forward - FORWARD_OFFSET) / METERS_PER_PIXEL))


def draw_segments(image: np.ndarray, starts: np.ndarray, ends: np.ndarray, radii: np.ndarray | float, intensity: int,
                  *, overlay: bool = False, caps: np.ndarray | None = None):
  """Strokes with round joins and one pixel of analytic edge antialiasing.

  caps is (segments, 2): whether each segment's start/end gets a round cap; without it every end is round.
  Max blending (the default) makes draw order irrelevant; overlay replaces the pixels beneath.
  """
  radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(starts),))
  reach = radii[:, None] + 1
  low = np.maximum(np.floor(np.minimum(starts, ends) - reach), 0).astype(int)
  high = np.minimum(np.ceil(np.maximum(starts, ends) + reach), [image.shape[1] - 1, image.shape[0] - 1]).astype(int)
  # Cull offscreen segments in one pass, then loop over plain Python floats.
  visible = np.flatnonzero((high >= low).all(axis=1))
  round_caps = np.ones((len(starts), 2), dtype=bool) if caps is None else caps
  for i, (sx, sy), (ex, ey), (x0, y0), (x1, y1), radius in zip(visible.tolist(), starts[visible].tolist(), ends[visible].tolist(),
                                                               low[visible].tolist(), high[visible].tolist(), radii[visible].tolist(),
                                                               strict=True):
    dx, dy = ex - sx, ey - sy
    px = np.arange(x0, x1 + 1) + (0.5 - sx)
    py = (np.arange(y0, y1 + 1) + (0.5 - sy))[:, None]
    along = (px * dx + py * dy) / max(dx * dx + dy * dy, 1e-12)
    t = np.clip(along, 0, 1)
    alpha = np.clip(radius + 0.5 - np.hypot(px - t * dx, py - t * dy), 0, 1)
    if not round_caps[i, 0]:
      alpha = alpha * (along >= 0)
    if not round_caps[i, 1]:
      alpha = alpha * (along <= 1)
    view = image[y0:y1 + 1, x0:x1 + 1]
    if overlay:
      view[:] = np.rint(view * (1 - alpha) + intensity * alpha).astype(np.uint8)
    else:
      np.maximum(view, np.rint(alpha * intensity).astype(np.uint8), out=view)


def draw_line(image: np.ndarray, points: np.ndarray, intensity: int, width: float, *, overlay: bool = False):
  draw_segments(image, points[:-1], points[1:], np.full(len(points) - 1, width / 2.0), intensity, overlay=overlay)


def draw_vehicle(image: np.ndarray, center: np.ndarray):
  """The training ego marker: a white arrow with a black outline, drawn over everything else."""
  arrow = ARROW + center
  x0, y0 = np.maximum(np.floor(arrow.min(axis=0)) - 2, 0).astype(int)
  x1, y1 = np.minimum(np.ceil(arrow.max(axis=0)) + 2, [image.shape[1], image.shape[0]]).astype(int)
  if x1 <= x0 or y1 <= y0:
    return
  px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
  inside = np.zeros(px.shape, dtype=bool)
  for a, b, c in ((arrow[0], arrow[1], arrow[2]), (arrow[0], arrow[2], arrow[3])):  # the concave arrow as two triangles
    sides = [(q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0]) for p, q in ((a, b), (b, c), (c, a))]
    inside |= ((sides[0] >= 0) & (sides[1] >= 0) & (sides[2] >= 0)) | ((sides[0] <= 0) & (sides[1] <= 0) & (sides[2] <= 0))
  image[y0:y1, x0:x1][inside] = 255
  draw_segments(image, arrow, np.roll(arrow, -1, axis=0), ARROW_OUTLINE / 2, 0, overlay=True)


class MapRenderer:
  def __init__(self, archive: PMTiles):
    self.archive = archive
    self.zoom = min(max(ZOOM, archive.min_zoom), archive.max_zoom)
    self.roads = lru_cache(maxsize=128)(self._load_roads)

  def _load_roads(self, x: int, y: int):
    """A tile's visible road segments as flat arrays: mercator starts/ends, style index, level and polyline end flags."""
    tile = self.archive.get(self.zoom, x, y)
    if tile is None:
      return None
    starts, ends, styles, levels, caps = [], [], [], [], []
    for road in decode_roads(tile):
      style = road_style(road)
      if style is None:
        continue
      points = (road.points / road.extent + np.array([x, y])) / (1 << self.zoom)
      n = len(points) - 1
      starts.append(points[:-1])
      ends.append(points[1:])
      styles.append(np.full(n, STYLES.index(style)))
      levels.append(np.full(n, road_level(road)))
      ends_of_line = np.zeros((n, 2), dtype=bool)
      ends_of_line[0, 0] = ends_of_line[-1, 1] = True
      caps.append(ends_of_line)
    if not starts:
      return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty((0, 2), dtype=bool)
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(styles), np.concatenate(levels), np.concatenate(caps)

  def view_tiles(self, longitude: float, latitude: float) -> tuple[list[tuple[int, int]], bool]:
    """Tiles covering the view at a point for every heading, and whether the view stays within the map."""
    origin = mercator(np.array([[longitude, latitude]]))[0]
    # A bounding circle includes all headings, forward offset and road stroke widths.
    radius = (SIZE * METERS_PER_PIXEL / math.sqrt(2) + FORWARD_OFFSET + 16) / (EARTH_CIRCUMFERENCE * math.cos(math.radians(latitude)))
    count = 1 << self.zoom
    low = np.floor((origin - radius) * count).astype(int)
    high = np.floor((origin + radius) * count).astype(int)
    if (high[0] - low[0] + 1) * (high[1] - low[1] + 1) > 256:
      raise ValueError("View requires more than 256 tiles; use a lower zoom")
    tiles = [(x % count, y) for y in range(low[1], high[1] + 1) for x in range(low[0], high[0] + 1) if 0 <= y < count]
    return tiles, len(tiles) == (high[0] - low[0] + 1) * (high[1] - low[1] + 1)

  def route_tiles(self, route: np.ndarray) -> set[tuple[int, int]]:
    """Every tile the view can show while the pose is snapped onto the route."""
    tiles: set[tuple[int, int]] = set()
    for a, b in zip(route[:-1], route[1:], strict=True):
      meters = np.hypot(*((b - a) * [111320.0 * math.cos(math.radians(a[1])), 110540.0]))
      for t in np.linspace(0, 1, int(meters / 100) + 2):
        tiles.update(self.view_tiles(*(a + (b - a) * t))[0])
    return tiles

  def prefetch(self, route: np.ndarray):
    self.archive.prefetch((self.zoom, x, y) for x, y in self.route_tiles(route))

  def render(self, position: Position, route: np.ndarray | None = None) -> RenderResult:
    image = np.full((SIZE, SIZE), BACKGROUND, dtype=np.uint8)
    wanted, complete = self.view_tiles(position.longitude, position.latitude)
    loaded = [self.roads(x, y) for x, y in wanted]
    tiles = [tile for tile in loaded if tile is not None]
    complete = complete and len(tiles) == len(loaded)
    # The training renderer's latitude-dependent style zoom: the web mercator zoom with METERS_PER_PIXEL
    # at this latitude, under the 512-pixel tile convention. Tile zoom and style zoom are independent.
    style_zoom = math.log2(math.cos(math.radians(position.latitude)) * EARTH_CIRCUMFERENCE / (256 * METERS_PER_PIXEL)) - 1
    ego = np.array([SIZE / 2, SIZE / 2 + FORWARD_OFFSET / METERS_PER_PIXEL])
    road_distance = math.inf
    if tiles:
      starts, ends, styles, levels, caps = (np.concatenate(column) for column in zip(*tiles, strict=True))
      starts, ends = project(starts, position), project(ends, position)
      # Tunnels are drawn at their interior width; everywhere else the white casing widens the stroke.
      table = np.array([[interpolate_width(style_zoom, *style.interior), interpolate_width(style_zoom, *style.casing)]
                        for style in STYLES])
      widths = table[styles, 0] + np.where(levels < 0, 0, 2 * table[styles, 1])
      road_distance = nearest_distance(ego, starts, ends) * METERS_PER_PIXEL
      for level in np.unique(levels):
        chosen = levels == level
        if level > levels.min():
          # Cut the roads beneath, open at way endpoints so a bridge does not end in a black oval.
          draw_segments(image, starts[chosen], ends[chosen], (widths[chosen] + CUT_WIDTH) / 2, BACKGROUND,
                        overlay=True, caps=~caps[chosen])
        draw_segments(image, starts[chosen], ends[chosen], widths[chosen] / 2, ROAD_INTENSITY)
    route_present = route is not None and len(route) >= 2
    if route_present:
      draw_line(image, project(mercator(route), position), ROUTE_INTENSITY, ROUTE_WIDTH, overlay=True)
    draw_vehicle(image, ego)
    return RenderResult(image, complete, route_present, road_distance)


def nearest_distance(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> float:
  """Distance from a point to the closest of the segments."""
  if not len(starts):
    return math.inf
  delta = ends - starts
  length2 = np.maximum((delta * delta).sum(axis=1), 1e-12)
  t = np.clip(((point - starts) * delta).sum(axis=1) / length2, 0, 1)
  return float(np.linalg.norm(starts + t[:, None] * delta - point, axis=1).min())
