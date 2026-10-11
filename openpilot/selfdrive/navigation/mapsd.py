#!/usr/bin/env python3
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import SIZE, MapRenderer, Position
from openpilot.selfdrive.navigation.route import RouteTracker, route_coordinates
from openpilot.selfdrive.navigation.tiles import RemoteTiles

RETRY_SECONDS = 10


def main():
  rk = Ratekeeper(20.0)
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navRoute'])
  server = VisionIpcServer('mapsd')
  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
  server.start_listener()
  archive = renderer = tracker = None
  route = np.empty((0, 2))
  prefetched = False
  retry_at = 0.0
  try:
    while True:
      sm.update(0)
      if sm.updated['navRoute']:
        new_route = route_coordinates(sm['navRoute']) if sm.valid['navRoute'] else np.empty((0, 2))
        if not np.array_equal(new_route, route):
          route = new_route
          tracker = RouteTracker(route) if len(route) >= 2 else None  # progress along the route starts over
          prefetched = False
      now = time.monotonic()
      source = latest_gps_source(sm)
      image = np.zeros((SIZE, SIZE), dtype=np.uint8)
      timestamp = 0
      try:
        if tracker is not None and now >= retry_at:
          if renderer is None:
            archive = PMTiles(RemoteTiles())
            renderer = MapRenderer(archive)
          if not prefetched:
            # TODO: fetch the tiles more intelligently on demand
            # Every tile the route can show is fetched up front, so each render below is a cache hit.
            renderer.prefetch(route)
            prefetched = True
          if source is not None:
            gps = sm[source]
            # As in training, the map is centered on the route and headed along it, not at the GPS fix.
            longitude, latitude, bearing = tracker.snap(gps.longitude, gps.latitude)
            result = renderer.render(Position(latitude, longitude, bearing), route)
            if result.valid:
              image, timestamp = result.image, sm.logMonoTime[source]
      except (OSError, ValueError):
        cloudlog.exception('mapsd.render_failed')
        if archive is not None:
          archive.close()
        archive = renderer = None
        prefetched = False
        retry_at = now + RETRY_SECONDS
      # A frame without a usable map is black rather than stale.
      server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), rk.frame, timestamp, time.monotonic_ns())
      rk.keep_time()
  finally:
    if archive is not None:
      archive.close()


if __name__ == '__main__':
  main()
