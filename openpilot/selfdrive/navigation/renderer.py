"""Deterministic CPU rasterization of Protomaps roads and a supplied route.

Styled after the training map renderer: 1 m per pixel, the ego three quarters
down the frame, pure white roads whose casings are part of the stroke, bridges cutting over the roads beneath
them, a grey route and a white ego arrow.
"""
import math
from functools import lru_cache

import numpy as np

from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.route import cumulative_distance, nearest_segment
from openpilot.selfdrive.navigation.vector_tile import Road, decode_roads

SIZE = 256
ZOOM = 15  # tile zoom; the style zoom is derived from the 1 m per pixel scale instead
FORWARD_OFFSET = 64.0  # the ego sits three quarters down the frame
EGO = np.array([SIZE / 2, SIZE / 2 + FORWARD_OFFSET])  # pixel position of the ego
EARTH_CIRCUMFERENCE = 40075016.68557849  # equatorial, for Web Mercator
BACKGROUND = 0
ROAD_INTENSITY = 255
ROUTE_INTENSITY = 128
ROUTE_WIDTH = 5.0  # pixels
CUT_WIDTH = 3.0  # black separator under a bridge, added to the road width
MAX_ROAD_DISTANCE = 100.0  # meters from the ego to the nearest road centerline for a valid map
ARROW = np.array([(0, -9), (7, 7), (0, 3), (-7, 7)], dtype=np.float64)  # ego marker, pixels around the ego
ARROW_OUTLINE = 2.0  # pixels


# The training style's road classes as (zoom, width, zoom, width) stops for the interior and for one side of the
# casing, in pixels. Casings are white there too, so they only widen the stroke. Only each class's top interval is
# kept: the style zoom at 1 m per pixel stays above 14 until 78 degrees of latitude, where the lower stops begin.
MAJOR, MEDIUM, STREET, MINOR, TRACK = range(5)
STOPS = np.array([[[5, 0.75, 18, 32], [10, 1, 18, 2]],  # MAJOR: motorway, trunk, primary
                  [[5, 0.1, 18, 26], [10, 0.75, 18, 2]],  # MEDIUM: secondary, tertiary
                  [[14, 2, 18, 18], [12, 0.75, 20, 2]],  # STREET: residential, unclassified, living streets, major links
                  [[14, 0.5, 18, 12], [12, 0.75, 20, 2]],  # MINOR: service roads and minor links
                  [[14, 1, 18, 12], [12, 0.75, 20, 2]]], dtype=np.float64)  # TRACK


def stroke_widths(zoom: float, styles: np.ndarray, levels: np.ndarray) -> np.ndarray:
  """Full stroke width of each segment: the style's exponential (base 1.5) interpolation between its stops, flat
  beyond them, with the casing on both sides except for tunnels, which are drawn at their interior width."""
  z0, w0, z1, w1 = np.moveaxis(STOPS, -1, 0)
  z = np.clip(zoom, z0, z1)
  table = w0 + (w1 - w0) * (1.5 ** (z - z0) - 1) / (1.5 ** (z1 - z0) - 1)  # (style, interior / casing)
  return table[styles, 0] + np.where(levels < 0, 0, 2 * table[styles, 1])


def road_style(road: Road) -> int | None:
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


def mercator(coordinates: np.ndarray) -> np.ndarray:
  """Longitude/latitude -> normalized Web Mercator, x east / y south."""
  longitude, latitude = coordinates.T
  y = (1.0 - np.arcsinh(np.tan(np.deg2rad(latitude))) / np.pi) / 2.0
  return np.column_stack(((longitude + 180.0) / 360.0, y))


def project(points: np.ndarray, longitude: float, latitude: float, bearing: float) -> np.ndarray:
  """Mercator points to pixels of a frame at a position, headed along a bearing in degrees clockwise from north."""
  origin = mercator(np.array([[longitude, latitude]]))[0]
  delta = points - origin
  delta[:, 0] = (delta[:, 0] + 0.5) % 1.0 - 0.5  # wrap across the antimeridian
  delta *= EARTH_CIRCUMFERENCE * math.cos(math.radians(latitude))  # meters, which are pixels
  c, s = math.cos(math.radians(bearing)), math.sin(math.radians(bearing))
  return EGO + delta @ np.array([[c, -s], [s, c]])  # x right, y down


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


def draw_vehicle(image: np.ndarray):
  """The training ego marker: a white arrow with a black outline, drawn over everything else."""
  arrow = ARROW + EGO
  (x0, y0), (x1, y1) = (np.floor(arrow.min(axis=0)) - 2).astype(int), (np.ceil(arrow.max(axis=0)) + 2).astype(int)
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
    starts, ends, styles, levels, caps = [np.empty((0, 2))], [np.empty((0, 2))], [np.empty(0, int)], [np.empty(0, int)], [np.empty((0, 2), bool)]
    for road in decode_roads(tile):
      style = road_style(road)
      if style is None:
        continue
      points = (road.points + np.array([x, y])) / (1 << self.zoom)
      n = len(points) - 1
      starts.append(points[:-1])
      ends.append(points[1:])
      styles.append(np.full(n, style))
      levels.append(np.full(n, road_level(road)))
      ends_of_line = np.zeros((n, 2), dtype=bool)
      ends_of_line[0, 0] = ends_of_line[-1, 1] = True
      caps.append(ends_of_line)
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(styles), np.concatenate(levels), np.concatenate(caps)

  def view_tiles(self, longitude: float, latitude: float) -> list[tuple[int, int]]:
    """Tiles covering the view at a point for every heading."""
    origin = mercator(np.array([[longitude, latitude]]))[0]
    # A bounding circle includes all headings, forward offset and road stroke widths.
    radius = (SIZE / math.sqrt(2) + FORWARD_OFFSET + 16) / (EARTH_CIRCUMFERENCE * math.cos(math.radians(latitude)))
    count = 1 << self.zoom
    low = np.floor((origin - radius) * count).astype(int)
    high = np.floor((origin + radius) * count).astype(int)
    return [(x % count, y) for y in range(low[1], high[1] + 1) for x in range(low[0], high[0] + 1) if 0 <= y < count]

  def route_tiles(self, route: np.ndarray) -> set[tuple[int, int]]:
    """Every tile the view can show while the pose is snapped onto the route, sampled every 100 m."""
    distance = cumulative_distance(route)
    samples = np.column_stack([np.interp(np.arange(0, distance[-1], 100), distance, route[:, axis]) for axis in (0, 1)])
    return {tile for lon, lat in np.vstack((samples, route)).tolist() for tile in self.view_tiles(lon, lat)}

  def prefetch(self, route: np.ndarray):
    self.archive.prefetch((self.zoom, x, y) for x, y in self.route_tiles(route))

  def render(self, longitude: float, latitude: float, bearing: float, route: np.ndarray) -> tuple[np.ndarray, bool]:
    """The map image at a position and bearing, and whether every tile was present with a road near the ego."""
    image = np.full((SIZE, SIZE), BACKGROUND, dtype=np.uint8)
    loaded = [self.roads(x, y) for x, y in self.view_tiles(longitude, latitude)]
    tiles = [tile for tile in loaded if tile is not None]
    # The training renderer's latitude-dependent style zoom: the web mercator zoom with one meter per pixel
    # at this latitude, under the 512-pixel tile convention. Tile zoom and style zoom are independent.
    style_zoom = math.log2(math.cos(math.radians(latitude)) * EARTH_CIRCUMFERENCE / 256) - 1
    road_distance = math.inf
    if tiles:
      starts, ends, styles, levels, caps = (np.concatenate(column) for column in zip(*tiles, strict=True))
      starts, ends = project(starts, longitude, latitude, bearing), project(ends, longitude, latitude, bearing)
      widths = stroke_widths(style_zoom, styles, levels)
      if len(starts):
        road_distance = nearest_segment(starts, ends, EGO)[2]
      for level in np.unique(levels):
        chosen = levels == level
        if level > levels.min():
          # Cut the roads beneath, open at way endpoints so a bridge does not end in a black oval.
          draw_segments(image, starts[chosen], ends[chosen], (widths[chosen] + CUT_WIDTH) / 2, BACKGROUND,
                        overlay=True, caps=~caps[chosen])
        draw_segments(image, starts[chosen], ends[chosen], widths[chosen] / 2, ROAD_INTENSITY)
    pixels = project(mercator(route), longitude, latitude, bearing)
    draw_segments(image, pixels[:-1], pixels[1:], ROUTE_WIDTH / 2, ROUTE_INTENSITY, overlay=True)
    draw_vehicle(image)
    return image, len(tiles) == len(loaded) and road_distance <= MAX_ROAD_DISTANCE
