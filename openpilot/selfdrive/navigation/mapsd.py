#!/usr/bin/env python3
"""Render Protomaps tiles to map images, without a window or graphics context."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import SIZE, MapRenderer, Position, RenderResult
from openpilot.selfdrive.navigation.route import RouteTracker, route_coordinates


class Mapsd:
  def __init__(self, renderer: MapRenderer):
    self.renderer = renderer
    self.route = np.empty((0, 2))
    self.tracker: RouteTracker | None = None

  def set_route(self, route: np.ndarray):
    """(longitude, latitude) rows. Progress along the route starts over."""
    self.route = route.reshape((-1, 2))
    self.tracker = RouteTracker(self.route) if len(self.route) >= 2 else None

  def update(self, position: Position) -> RenderResult:
    # As in training, the map is centered on the route and headed along it, not at the GPS fix.
    if self.tracker is not None:
      longitude, latitude, bearing = self.tracker.snap(position.longitude, position.latitude)
      position = Position(latitude, longitude, bearing)
    return self.renderer.render(position, self.route)


def main():
  rk = Ratekeeper(20.0)
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navRoute'])
  archive = mapsd = None
  rendered_revision = -1

  def render(position, route, revision):
    nonlocal archive, mapsd, rendered_revision
    try:
      if mapsd is None:
        archive = PMTiles()
        mapsd = Mapsd(MapRenderer(archive))
      if revision != rendered_revision:
        mapsd.set_route(route)
        rendered_revision = revision
      return mapsd.update(position)
    except (OSError, ValueError):
      if archive is not None:
        archive.close()
      archive = mapsd = None
      rendered_revision = -1
      raise

  route = np.empty((0, 2))
  revision = 0
  pending = None
  retry_at = 0.0
  result = None
  received_at = timestamp = 0
  server = VisionIpcServer('mapsd')
  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
  server.start_listener()
  try:
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='mapsd-render') as worker:
      while True:
        sm.update(0)
        if sm.updated['navRoute']:
          new_route = route_coordinates(sm['navRoute']) if sm.valid['navRoute'] else np.empty((0, 2))
          if not np.array_equal(new_route, route):
            route = new_route
            revision += 1
            result = None
        now = time.monotonic()
        source = latest_gps_source(sm)
        if pending is not None and pending[0].done():
          future, rendered_route, frame_timestamp, received = pending
          try:
            frame = future.result()
            if rendered_route == revision:
              result, received_at, timestamp = frame, received, frame_timestamp
          except (OSError, ValueError):
            logging.exception('Unable to render map')
            result = None
            retry_at = now + 10
          pending = None
        if pending is None and source is not None and len(route) >= 2 and now >= retry_at:
          gps = sm[source]
          position = Position(gps.latitude, gps.longitude, gps.bearingDeg)
          pending = worker.submit(render, position, route, revision), revision, sm.logMonoTime[source], sm.recv_time[source]
        # Keep publishing black frames during tile fetching, GPS loss or route changes.
        valid = source is not None and result is not None and result.valid and 0 <= now - received_at < 2
        image = result.image if valid else np.zeros((SIZE, SIZE), dtype=np.uint8)
        server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), rk.frame, timestamp if valid else 0, time.monotonic_ns())
        rk.keep_time()
  finally:
    if archive is not None:
      archive.close()


if __name__ == '__main__':
  main()
