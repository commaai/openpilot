import gzip
import json
import re
import struct
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

import openpilot.cereal.messaging as messaging
from openpilot.common.prefix import OpenpilotPrefix
from msgq.visionipc import VisionIpcClient
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.selfdrive.navigation.helpers import Coordinate
from openpilot.selfdrive.navigation.mapsd import main as mapsd_main
from openpilot.selfdrive.navigation.navd import REROUTE_COUNTER_MIN, RouteEngine
from openpilot.selfdrive.navigation.pmtiles import PMTiles, decode_directory, tile_id
from openpilot.selfdrive.navigation.renderer import SIZE, MapRenderer, Position, draw_line, mercator, project
from openpilot.selfdrive.navigation.route import RouteTracker
from openpilot.selfdrive.navigation.vector_tile import decode_lines, decode_roads

METER = 1 / 111319.49079327358  # degrees of latitude


def varint(value):
  output = bytearray()
  while value > 127:
    output.append((value & 127) | 128)
    value >>= 7
  output.append(value)
  return bytes(output)


def field(number, value):
  if isinstance(value, bytes):
    return varint(number << 3 | 2) + varint(len(value)) + value
  return varint(number << 3) + varint(value)


def road_tile(crossing=None):
  # One vertical line, from (2048, 0) to (2048, 4096), and one non-road layer. With crossing='bridge' or 'tunnel',
  # a horizontal line from (1024, 2048) to (3072, 2048) at that level.
  geometry = b''.join(varint(v) for v in [9, 4096, 0, 10, 0, 8192])
  feature = field(2, b'\x00\x00') + field(3, 2) + field(4, geometry)
  roads = field(15, 2) + field(1, b'roads') + field(2, feature)
  if crossing is not None:
    geometry = b''.join(varint(v) for v in [9, 2048, 4096, 10, 4096, 0])
    roads += field(2, field(2, b'\x00\x00\x01\x01') + field(3, 2) + field(4, geometry))
  roads += field(3, b'kind') + field(3, f'is_{crossing}'.encode()) + field(4, field(1, b'minor_road')) + field(4, field(7, 1)) + field(5, 4096)
  return field(3, field(1, b'buildings')) + field(3, roads)


def make_archive(payload=None, leaf=False, compression=2):
  """A PMTiles archive with one zoom-zero tile, in the root directory or in a leaf directory."""
  compress = gzip.compress if compression == 2 else lambda data: data
  tile = compress(road_tile() if payload is None else payload)
  directory = compress(varint(1) + varint(0) + varint(1) + varint(len(tile)) + varint(1))
  child = directory if leaf else b''
  root = compress(varint(1) + varint(0) + varint(0) + varint(len(child)) + varint(1)) if leaf else directory
  metadata = compress(b'{"vector_layers":[{"id":"roads"}]}')
  header = bytearray(127)
  header[:8] = b'PMTiles\x03'
  offsets = [127, 127 + len(root), 127 + len(root) + len(metadata), 127 + len(root) + len(metadata) + len(child)]
  struct.pack_into('<11Q', header, 8, offsets[0], len(root), offsets[1], len(metadata), offsets[2], len(child), offsets[3], len(tile), 1, 1, 1)
  header[96:102] = bytes([1, compression, compression, 1, 0, 0])
  return bytes(header + root + metadata + child + tile)


@contextmanager
def local_archive(data):
  """A PMTiles reader over in-memory bytes, bypassing the tile server."""
  source = MagicMock(size=len(data))
  source.read.side_effect = lambda offset, length: data[offset:offset + length]
  with patch('openpilot.selfdrive.navigation.pmtiles.RemoteTiles', return_value=source), PMTiles() as archive:
    yield archive


def mapbox_response(steps):
  response = MagicMock(status_code=200)
  response.json.side_effect = lambda: {'routes': [{'legs': [{'steps': steps}]}]}
  return response


def route_engine(sm, pm, values):
  params = MagicMock()
  params.get.side_effect = values.get
  params.remove.side_effect = lambda key: values.pop(key, None)
  with patch('openpilot.selfdrive.navigation.navd.Api') as api:
    api.return_value.get_token.return_value = 'test-token'
    return RouteEngine(sm, pm, params=params)


class RangeHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    self.server.requests.append(dict(self.headers))
    if self.headers.get('If-Match', self.server.etag) != self.server.etag:
      self.send_error(412)
      return
    if self.server.ignore_range:
      self.send_response(200)
      self.end_headers()
      return
    offset, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers['Range']).groups())
    data = self.server.archive[offset:end + 1]
    self.send_response(206)
    self.send_header('Content-Range', f'bytes {offset}-{end}/{len(self.server.archive)}')
    self.send_header('Content-Length', str(len(data)))
    self.send_header('ETag', self.server.etag)
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, *args):
    pass


class TileServerCase(unittest.TestCase):
  """Serves a leaf-directory archive over HTTP range requests, with the comma API and tile cache stubbed."""

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.cache = Path(self.directory.name) / 'navigation' / 'tiles'
    self.server = ThreadingHTTPServer(('127.0.0.1', 0), RangeHandler)
    self.server.archive = make_archive(leaf=True)
    self.server.etag = '"test-version"'
    self.server.ignore_range = False
    self.server.requests = []
    threading.Thread(target=self.server.serve_forever, daemon=True).start()
    self.addCleanup(self.server.server_close)
    self.addCleanup(self.server.shutdown)
    self.api = MagicMock(dongle_id='test-dongle')
    self.api.get_token.return_value = 'device-token'
    self.url = f'http://127.0.0.1:{self.server.server_port}/planet.pmtiles?sig=secret'
    self.api.get.return_value.json.return_value = {'url': self.url, 'expires': time.time() + 86400}  # noqa: TID251
    for name, value in [('Api', MagicMock(return_value=self.api)), ('Params', MagicMock()), ('PC', True)]:
      self.enterContext(patch(f'openpilot.selfdrive.navigation.tiles.{name}', value))
    self.enterContext(patch('openpilot.selfdrive.navigation.tiles.Paths.comma_home', return_value=self.directory.name))


class TestEndToEnd(TileServerCase):
  def test_navd_to_map_frames(self):
    """navd routes a destination over messaging; mapsd fetches tiles over HTTP and publishes map frames."""
    destination = {'latitude': 0.01, 'longitude': 0}
    self.enterContext(patch('requests.get', return_value=mapbox_response([
      {'geometry': {'coordinates': [[0, -0.01], [0, 0.01]]}, 'distance': 2224, 'duration': 100, 'bannerInstructions': []}])))
    with OpenpilotPrefix():
      pm = messaging.PubMaster(['navInstruction', 'navRoute', 'gpsLocationExternal'])
      values = {'LastGPSPosition': json.dumps({'latitude': 0, 'longitude': 0}), 'NavDestination': dict(destination)}
      navd = route_engine(messaging.SubMaster(['gpsLocation', 'gpsLocationExternal']), pm, values)
      state = {'step': 0, 'fix': True, 'deadline': time.monotonic() + 10, 'client': None}
      # Each step waits for a map or a black frame, then changes something. Messages take a loop to propagate.
      steps = [
        (False, lambda: (values.pop('NavDestination'), navd.update())),  # a map, then the destination is cleared
        (True, lambda: (values.update(NavDestination=dict(destination)), navd.update())),  # black, then a new destination
        (False, lambda: state.update(fix=False)),  # a map again, then the GPS fix is lost
        (True, None),  # black, not stale
      ]

      def on_frame():
        self.assertLess(time.monotonic(), state['deadline'], state['step'])
        if state['client'] is None:  # the server exists now
          state['client'] = client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
          while not client.connect(False) and time.monotonic() < state['deadline']:
            time.sleep(0.01)
          self.assertTrue(client.is_connected())
          navd.update()
          return
        client = state['client']
        gps = messaging.new_message('gpsLocationExternal', valid=True)
        gps.gpsLocationExternal.hasFix = state['fix']
        pm.send('gpsLocationExternal', gps)
        frame = client.recv(timeout_ms=1000)
        assert frame is not None
        self.assertEqual((frame.width, frame.height, frame.stride), (SIZE, SIZE, SIZE))
        image = np.frombuffer(bytes(frame.data), dtype=np.uint8).reshape((SIZE, SIZE))
        expect_black, action = steps[state['step']]
        if bool(image.any()) == expect_black:
          return
        if not expect_black:
          # The route is centred over the white road, with the ego arrow three quarters down.
          self.assertEqual(tuple(image[[128, 128, 188, 195], [128, 131, 128, 128]]), (128, 255, 255, 0))
          self.assertEqual(client.frame_id, 42)
          self.assertGreater(client.timestamp_sof, 0)
        state['step'] += 1
        if action is None:
          raise StopIteration
        action()

      rk = MagicMock(frame=42)
      rk.keep_time.side_effect = on_frame
      with patch('openpilot.selfdrive.navigation.mapsd.Ratekeeper', return_value=rk), self.assertRaises(StopIteration):
        mapsd_main()
      self.assertEqual(state['step'], len(steps))
      self.api.get.assert_called_once_with('v1/test-dongle/nav/tiles', access_token='device-token', timeout=10)
      self.assertTrue(all('Range' in headers for headers in self.server.requests))
      self.assertTrue(all(headers['If-Match'] == self.server.etag for headers in self.server.requests[1:]))
      self.assertTrue(all(b'secret' not in p.read_bytes() for p in self.cache.iterdir()))


class TestRemoteTiles(TileServerCase):
  def test_shared_cache_and_refresh(self):
    with PMTiles() as archive:
      archive.remote.expires = 0  # the signed URL is refreshed before the next fetch
      self.assertEqual(archive.get(0, 0, 0), road_tile())
      self.assertEqual(archive.get(0, 0, 0), road_tile())
      self.assertEqual(self.api.get.call_count, 2)
    count = len(self.server.requests)
    with PMTiles() as archive:
      self.assertEqual(archive.get(0, 0, 0), road_tile())
    # A second reader revalidates only the header; directories and tiles are shared on disk.
    self.assertEqual(len(self.server.requests), count + 1)
    with patch('openpilot.selfdrive.navigation.tiles.CACHE_LIMIT', 1), PMTiles() as archive:
      archive.remote.read(1, 2)  # an uncached range is downloaded, then the cache is trimmed to the limit
    self.assertEqual(list(self.cache.iterdir()), [])

  def test_failures(self):
    with PMTiles() as archive:
      self.server.etag = '"new-version"'
      with self.assertRaisesRegex(OSError, '412'):  # the archive changed under an open reader
        archive.get(0, 0, 0)
    with PMTiles() as archive:
      self.assertEqual(archive.get(0, 0, 0), road_tile())
    self.server.ignore_range = True
    with self.assertRaisesRegex(OSError, '200'):
      PMTiles()
    self.api.get.return_value.raise_for_status.side_effect = requests.HTTPError(self.url)
    with self.assertRaisesRegex(OSError, '^Unable to fetch map tiles$'):  # the signed URL stays out of the error
      PMTiles()


class TestRouteEngine(unittest.TestCase):
  def setUp(self):
    self.pm = MagicMock()
    self.values = {'LastGPSPosition': json.dumps({'latitude': 0, 'longitude': 0}), 'NavDestination': {'latitude': 0, 'longitude': 0.002},
                   'LanguageSetting': 'main_en'}
    self.sm = MagicMock()
    self.sm.valid = dict.fromkeys(('gpsLocation', 'gpsLocationExternal'), True)
    self.sm.logMonoTime = dict.fromkeys(self.sm.valid, time.monotonic_ns())
    self.sm.recv_time = dict.fromkeys(self.sm.valid, time.monotonic())
    self.sm.__getitem__.return_value = SimpleNamespace(hasFix=True, latitude=0, longitude=0, bearingDeg=90)
    self.engine = route_engine(self.sm, self.pm, self.values)
    self.engine.update_location()
    distance = Coordinate(0, 0).distance_to(Coordinate(0, 0.001))
    self.response = mapbox_response([
      {'geometry': {'coordinates': [[0, 0], [0.001, 0]]}, 'distance': distance, 'duration': 10,
       'bannerInstructions': [{'distanceAlongGeometry': 100, 'primary': {'text': 'Turn right', 'type': 'turn', 'modifier': 'right'}}]},
      {'geometry': {'coordinates': [[0.001, 0], [0.002, 0]]}, 'distance': distance, 'duration': 10, 'bannerInstructions': []},
    ])

  def last_message(self, service):
    return next(call.args[1] for call in reversed(self.pm.send.call_args_list) if call.args[0] == service)

  def load_route(self):
    with patch('requests.get', return_value=self.response) as get:
      self.engine.calculate_route(Coordinate(0, 0.002))
    self.engine.send_route()
    return get

  def test_request_and_instruction(self):
    get = self.load_route()
    self.assertEqual(get.call_args.args[0], 'https://api.commadotai.com/v1/nav/directions/v5/mapbox/driving-traffic/0,0;0.002,0')
    options = get.call_args.kwargs['params']
    self.assertEqual((options['access_token'], options['bearings'], options['waypoints'], options['language']), ('test-token', '90,90;', '0;1', 'en'))
    self.assertEqual(len(self.last_message('navRoute').navRoute.coordinates), 4)
    self.engine.send_instruction()
    message = self.last_message('navInstruction')
    self.assertTrue(message.valid)
    instruction = message.navInstruction.to_dict()
    self.assertEqual((instruction['maneuverPrimaryText'], instruction['maneuverModifier'], instruction['timeRemaining']), ('Turn right', 'right', 20))

  def test_maneuvers_and_rerouting(self):
    self.load_route()
    self.engine.last_position = Coordinate(0.001, 0)  # off route: reroute after a few consecutive misses
    for _ in range(REROUTE_COUNTER_MIN):
      self.assertFalse(self.engine.should_recompute())
    self.assertTrue(self.engine.should_recompute())
    self.engine.gps_ok = False  # but not while the GPS drifts in a tunnel
    with patch('requests.get') as get:
      self.engine.recompute_route()
      get.assert_not_called()
    self.engine.gps_ok = True
    self.engine.reset_recompute_limits()
    with patch('requests.get', return_value=self.response) as get:
      self.engine.recompute_route()
      self.assertEqual(get.call_count, 1)
    self.assertEqual((self.engine.recompute_backoff, self.engine.recompute_countdown), (1, 1))
    self.engine.last_position = Coordinate(0, 0.0012)  # past the first maneuver
    self.engine.send_instruction()
    self.assertEqual((self.engine.step_idx, self.engine.recompute_backoff), (1, 0))
    self.engine.last_position = Coordinate(0, 0.0024)  # past the destination
    self.engine.send_instruction()
    self.assertIsNone(self.engine.route)
    self.assertNotIn('NavDestination', self.values)

  def test_failed_request_clears_route(self):
    self.load_route()
    self.response.raise_for_status.side_effect = requests.HTTPError('test failure')
    self.load_route()
    self.assertIsNone(self.engine.route)
    self.assertEqual(len(self.last_message('navRoute').navRoute.coordinates), 0)
    self.engine.send_instruction()
    self.assertFalse(self.last_message('navInstruction').valid)

  def test_gps_source(self):
    self.sm.__getitem__.return_value = SimpleNamespace(hasFix=True, latitude=1, longitude=1, bearingDeg=180)
    self.sm.logMonoTime = dict.fromkeys(self.sm.valid, 1)  # replayed logs are judged by receipt time
    self.engine.update_location()
    self.assertTrue(self.engine.gps_ok)
    self.assertEqual(self.engine.last_position, Coordinate(1, 1))
    self.sm.recv_time = dict.fromkeys(self.sm.valid, time.monotonic() - 3)
    self.sm.__getitem__.return_value = SimpleNamespace(hasFix=True, latitude=2, longitude=2, bearingDeg=0)
    self.engine.update_location()
    self.assertFalse(self.engine.gps_ok)
    self.assertEqual((self.engine.last_position, self.engine.last_bearing), (Coordinate(1, 1), 180))


class TestTiles(unittest.TestCase):
  def test_pmtiles_format(self):
    self.assertEqual([tile_id(1, x, y) for x, y in [(0, 0), (0, 1), (1, 1), (1, 0)]], [1, 2, 3, 4])
    self.assertEqual(sorted(tile_id(3, x, y) for x in range(8) for y in range(8)), list(range(21, 85)))
    # IDs 5, 10; runs 3, 1; lengths 4, 6; the second offset follows the first blob.
    data = bytes([2, 5, 5, 3, 1, 4, 6, 1, 0])
    self.assertEqual(decode_directory(data), ([5, 10], [(3, 4, 0), (1, 6, 4)]))
    for invalid in (b'\x80', b'\x01\x00\x01\x01\x00', data + b'\x00'):
      with self.assertRaises(ValueError):
        decode_directory(invalid)
    for leaf in (False, True):
      for compression in (1, 2):
        with local_archive(make_archive(leaf=leaf, compression=compression)) as archive:
          self.assertEqual(archive.get(0, 0, 0), road_tile())
          self.assertIsNone(archive.get(1, 0, 0))
    data = bytearray(make_archive(compression=1))
    data[101] = 1  # max zoom
    data[129] = 3  # the zoom-zero tile is also addressed by IDs 1 and 2
    with local_archive(bytes(data)) as archive:
      self.assertEqual(archive.get(1, 0, 1), road_tile())
      self.assertIsNone(archive.get(1, 1, 1))
    with self.assertRaises(ValueError), local_archive(make_archive()[:-1]):
      pass

  def test_vector_tiles(self):
    roads = decode_roads(road_tile())
    self.assertEqual((len(roads), roads[0].kind, roads[0].extent), (1, 'minor_road', 4096))
    np.testing.assert_array_equal(roads[0].points, [[2048, 0], [2048, 4096]])
    lines = decode_lines([9, 20, 20, 10, 10, 0, 9, 9, 9, 10, 0, 10])
    np.testing.assert_array_equal(lines[0], [[10, 10], [15, 10]])
    np.testing.assert_array_equal(lines[1], [[10, 5], [10, 10]])
    for data in (b'\x1a\xff', b'\x00', b'\x1a\x02\x0a\x04'):
      with self.assertRaises(ValueError):
        decode_roads(data)
    for commands in ([10, 1, 1], [9, 0], [9, 0, 0, 15]):
      with self.assertRaises(ValueError):
        decode_lines(commands)


class TestRendering(unittest.TestCase):
  def test_projection(self):
    # At the equator, 64m ahead is the image center and the car is 64px below it, at 1m per pixel.
    points = mercator(np.array([[0, 0], [0, 64 * METER], [64 * METER, 0]]))
    np.testing.assert_allclose(project(points, Position(0, 0, 0)), [[128, 192], [128, 128], [192, 192]], atol=1e-5)
    np.testing.assert_allclose(project(points, Position(0, 0, 90)), [[128, 192], [64, 192], [128, 128]], atol=1e-5)
    pixels = project(mercator(np.array([[-179.999, 0]])), Position(0, 179.999, 90))  # across the antimeridian
    self.assertAlmostEqual(pixels[0, 1], 192 - 222.6389816, places=5)

  def test_strokes(self):
    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    draw_line(image, np.array([[-1000, 128], [1000, 128]]), 255, 8)
    self.assertTrue(np.all(image[128] == 255))
    draw_line(image, np.array([[128, 0], [128, 256]]), 128, 5, overlay=True)  # the grey route replaces the white road
    self.assertEqual((image[128, 128], image[128, 131]), (128, 255))
    before = image.copy()
    draw_line(image, np.array([[-1000, -1000], [-500, -500]]), 255, 5)
    np.testing.assert_array_equal(image, before)
    draw_line(image, np.array([[40, 40], [40, 40]]), 255, 5)  # a zero-length segment is still a dot
    self.assertEqual(image[40, 40], 255)

  def test_crossings(self):
    images = {}
    for crossing in ('bridge', 'tunnel'):
      with local_archive(make_archive(road_tile(crossing))) as archive:
        images[crossing] = MapRenderer(archive).render(Position(0, 0, 0), vehicle=False).image
    # The horizontal road crosses the vertical one at the ego, 192px down. A bridge severs the road beneath
    # just outside its own stroke; a tunnel passes underneath untouched. The cut has butt caps, so the
    # bridge's own end is not wrapped in black.
    self.assertEqual((images['bridge'][186, 128], images['bridge'][192, 128], images['bridge'][192, 200], images['bridge'][180, 128]), (0, 255, 255, 255))
    self.assertEqual((images['tunnel'][186, 128], images['tunnel'][192, 200]), (255, 255))

  def test_validity_and_display(self):
    route = np.array([[0, -0.01], [0, 0.01]])
    with local_archive(make_archive()) as archive:
      renderer = MapRenderer(archive)
      self.assertFalse(renderer.render(Position(0, 0, 0)).valid)  # no route
      result = renderer.render(Position(0, 0, 0), route)
      self.assertTrue(result.valid)
      self.assertEqual(result.road_distance, 0)
      far = renderer.render(Position(0, 0.01, 0), route)  # no road near the ego
      self.assertGreater(far.road_distance, 1000)
      self.assertFalse(far.valid)
      display = renderer.render(Position(0, 0, 0), route, width=490, height=254, route_color=(72, 235, 132))
      self.assertEqual(display.image.shape, (254, 490, 3))
      np.testing.assert_array_equal(display.image[-1, 245], [72, 235, 132])
      with patch.object(archive, 'get', return_value=None):
        missing = MapRenderer(archive).render(Position(0, 0, 0), route)
        self.assertFalse(missing.complete or missing.valid)
        self.assertEqual(missing.image[128, 128], 128)  # the route is still drawn


class TestRouteTracking(unittest.TestCase):
  def test_snap_heading_and_window(self):
    m = METER
    tracker = RouteTracker(np.array([[0, 0], [0, 200 * m], [200 * m, 200 * m]]))  # north 200m, then east 200m
    longitude, latitude, bearing = tracker.snap(3 * m, 100 * m)  # 3m right of the first leg
    self.assertAlmostEqual(tracker.progress, 100, delta=0.5)
    self.assertAlmostEqual(longitude, 0)  # lateral error removed
    self.assertAlmostEqual(latitude, 100 * m)
    self.assertAlmostEqual(bearing, 0, places=3)
    self.assertAlmostEqual(tracker.snap(0, 200 * m)[2], 45, places=2)  # at the corner the +-50m chord is mid-turn
    self.assertAlmostEqual(tracker.snap(0, 150 * m)[2], 45, places=2)  # progress never decreases
    self.assertAlmostEqual(tracker.snap(200 * m, 200 * m)[2], 90, places=2)  # the chord clips at the route's end
    # Out and back along the same road: a fix near the far leg stays on the outbound leg within the window,
    # while a fresh tracker searches anywhere and lands on the return leg.
    route = np.array([[0, 0], [0, 1000 * m], [10 * m, 1000 * m], [10 * m, 0]])
    tracker = RouteTracker(route)
    tracker.snap(1 * m, 100 * m)
    self.assertAlmostEqual(tracker.snap(9 * m, 150 * m)[2], 0, places=3)
    self.assertAlmostEqual(tracker.progress, 150, delta=0.5)
    self.assertAlmostEqual(RouteTracker(route).snap(9 * m, 150 * m)[2], 180, places=3)
    with self.assertRaises(ValueError):
      RouteTracker(np.array([[0, 0]])).snap(0, 0)


if __name__ == '__main__':
  unittest.main()
