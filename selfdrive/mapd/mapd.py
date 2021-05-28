#!/usr/bin/env python3
import threading
import numpy as np
from time import strftime, gmtime
import cereal.messaging as messaging
from common.realtime import Ratekeeper
from selfdrive.mapd.lib.osm import OSM
from selfdrive.mapd.lib.geo import distance_to_points
from selfdrive.mapd.lib.WayCollection import WayCollection
from .config import QUERY_RADIUS, MIN_DISTANCE_FOR_NEW_QUERY, FULL_STOP_MAX_SPEED


_DEBUG = False


def _debug(msg):
  if not _DEBUG:
    return
  print(msg)


def excepthook(args):
  _debug(f'MapD: Threading exception:\n{args}')


threading.excepthook = excepthook


class MapD():
  def __init__(self):
    self.osm = OSM()
    self.way_collection = None
    self.route = None
    self.last_gps_fix_timestamp = 0
    self.last_gps = None
    self.location_deg = None  # The current location in degrees.
    self.location_rad = None  # The current location in radians as a Numpy array.
    self.bearing_rad = None
    self.accuracy = None  # The current location accuracy in mts. 2 sigma.
    self.gps_speed = 0.
    self.last_fetch_location = None
    self.last_route_update_fix_timestamp = 0
    self.last_publish_fix_timestamp = 0
    self._query_thread = None
    self._lock = threading.RLock()

  def update_gps(self, sm):
    sock = 'gpsLocationExternal'
    if not sm.updated[sock] or not sm.valid[sock]:
      return

    log = sm[sock]
    self.last_gps = log

    # ignore the message if the fix is invalid
    if log.flags % 2 == 0:
      return

    self.last_gps_fix_timestamp = log.timestamp  # Unix TS. Milliseconds since January 1, 1970.
    self.location_rad = np.radians(np.array([log.latitude, log.longitude], dtype=float))
    self.location_deg = (log.latitude, log.longitude)
    self.bearing_rad = np.radians(log.bearingDeg, dtype=float)
    self.gps_speed = log.speed
    # log accuracies are presumably 1 sigma. To work with an accuracy that gives us >95% confidence and working
    # on the assumption that fix location follows a bell distribution we should use 2 * sigma instead.
    self.accuracy = log.accuracy * 2.

    _debug('Mapd: ********* Got GPS fix'
           f'Pos: {self.location_deg} +/- {self.accuracy} mts.\n'
           f'Bearing: {log.bearingDeg} +/- {log.bearingAccuracyDeg * 2.} deg.\n'
           f'timestamp: {strftime("%d-%m-%y %H:%M:%S", gmtime(self.last_gps_fix_timestamp * 1e-3))}'
           f'*******')

  def _query_osm_not_blocking(self):
    def query(osm, location_deg, location_rad, radius):
      _debug(f'Mapd: Start query for OSM map data at {location_deg}')
      lat, lon = location_deg
      ways = osm.fetch_road_ways_around_location(lat, lon, radius)
      _debug(f'Mapd: Query to OSM finished with {len(ways)} ways')

      # Only issue an update if we received some ways. Otherwise it is most likely a conectivity issue.
      # Will retry on next loop.
      if len(ways) > 0:
        new_way_collection = WayCollection(ways, location_rad)

        # Use the lock to update the way_collection as it might be being used to update the route.
        _debug('Mapd: Locking to write results from osm.')
        with self._lock:
          self.way_collection = new_way_collection
          self.last_fetch_location = location_rad
          _debug(f'Mapd: Updated map data @ {location_deg} - got {len(ways)} ways')

        _debug('Mapd: Releasing Lock to write results from osm')

    # Ignore if we have a query thread already running.
    if self._query_thread is not None and self._query_thread.is_alive():
      return

    self._query_thread = threading.Thread(target=query, args=(self.osm, self.location_deg, self.location_rad,
                                                              QUERY_RADIUS))
    self._query_thread.start()

  def updated_osm_data(self):
    if self.route is not None:
      distance_to_end = self.route.distance_to_end
      if distance_to_end is not None and distance_to_end >= MIN_DISTANCE_FOR_NEW_QUERY:
        # do not query as long as we have a route with enough distance ahead.
        return

    if self.location_rad is None:
      return

    if self.last_fetch_location is not None:
      distance_since_last = distance_to_points(self.last_fetch_location, np.array([self.location_rad]))[0]
      if distance_since_last < QUERY_RADIUS - MIN_DISTANCE_FOR_NEW_QUERY:
        # do not query if are still not close to the border of previous query area
        return

    self._query_osm_not_blocking()

  def update_route(self):
    def update_proc():
      if self.way_collection is None or self.location_rad is None or self.bearing_rad is None:
        _debug('Mapd *****: Can not update route. Missing WayCollection, location or bearing ********')
        return

      if self.last_route_update_fix_timestamp == self.last_gps_fix_timestamp:
        _debug('Mapd *****: Skipping route update. No new fix since last update ********')
        return

      self.last_route_update_fix_timestamp = self.last_gps_fix_timestamp

      # Create the route if not existent or if it was generated by an older way collection
      if self.route is None or self.route.way_collection_id != self.way_collection.id:
        self.route = self.way_collection.get_route(self.location_rad, self.bearing_rad, self.accuracy)
        _debug(f'Mapd *****: Route created: \n{self.route}\n********')
        return

      # Do not attempt to update the route if the car is going close to a full stop, as the bearing can start
      # jumping and creating unnecesary loosing of the route. Since the route update timestamp has been updated
      # a new liveMapData message will be published with the current values (which is desirable)
      if self.gps_speed < FULL_STOP_MAX_SPEED:
        _debug('Mapd *****: Route Not updated as car has Stopped ********')
        return

      self.route.update(self.location_rad, self.bearing_rad, self.accuracy)
      if self.route.located:
        _debug(f'Mapd *****: Route updated: \n{self.route}\n********')
        return

      # if an old route did not mange to locate, attempt to regenerate form way collection.
      self.route = self.way_collection.get_route(self.location_rad, self.bearing_rad, self.accuracy)
      _debug(f'Mapd *****: Failed to update location in route. Regenerated with route: \n{self.route}\n********')

    # We use the lock when updating the route, as it reads `way_collection` which can ben updated by
    # a new query result from the _query_thread.
    _debug('Mapd: Locking to update route.')
    with self._lock:
      update_proc()

    _debug('Mapd: Releasing Lock to update route')

  def publish(self, pm, sm):
    # Ensure we have a route currently located
    if self.route is None or not self.route.located:
      return

    # Ensure we have a route update since last publish
    if self.last_publish_fix_timestamp == self.last_route_update_fix_timestamp:
      return

    self.last_publish_fix_timestamp = self.last_route_update_fix_timestamp

    speed_limit = self.route.current_speed_limit
    next_speed_limit_section = self.route.next_speed_limit_section
    turn_speed_limit_section = self.route.current_curvature_speed_limit_section
    next_turn_speed_limit_section = self.route.next_curvature_speed_limit_section

    map_data_msg = messaging.new_message('liveMapData')
    map_data_msg.valid = sm.all_alive_and_valid(service_list=['gpsLocationExternal'])

    map_data_msg.liveMapData.lastGpsTimestamp = self.last_gps.timestamp
    map_data_msg.liveMapData.speedLimitValid = bool(speed_limit is not None)
    map_data_msg.liveMapData.speedLimit = float(speed_limit if speed_limit is not None else 0.0)
    map_data_msg.liveMapData.speedLimitAheadValid = bool(next_speed_limit_section is not None)
    map_data_msg.liveMapData.speedLimitAhead = float(next_speed_limit_section.value
                                                     if next_speed_limit_section is not None else 0.0)
    map_data_msg.liveMapData.speedLimitAheadDistance = float(next_speed_limit_section.start
                                                             if next_speed_limit_section is not None else 0.0)

    map_data_msg.liveMapData.turnSpeedLimitValid = bool(turn_speed_limit_section is not None)
    map_data_msg.liveMapData.turnSpeedLimit = float(turn_speed_limit_section.value
                                                    if turn_speed_limit_section is not None else 0.0)
    map_data_msg.liveMapData.turnSpeedLimitSign = int(turn_speed_limit_section.curv_sign
                                                      if turn_speed_limit_section is not None else 0)
    map_data_msg.liveMapData.turnSpeedLimitEndDistance = float(turn_speed_limit_section.end
                                                               if turn_speed_limit_section is not None else 0.0)
    map_data_msg.liveMapData.turnSpeedLimitAheadValid = bool(next_turn_speed_limit_section is not None)
    map_data_msg.liveMapData.turnSpeedLimitAhead = float(next_turn_speed_limit_section.value
                                                         if next_turn_speed_limit_section is not None else 0.0)
    map_data_msg.liveMapData.turnSpeedLimitAheadDistance = float(next_turn_speed_limit_section.start
                                                                 if next_turn_speed_limit_section is not None else 0.0)
    map_data_msg.liveMapData.turnSpeedLimitAheadSign = int(next_turn_speed_limit_section.curv_sign
                                                           if next_turn_speed_limit_section is not None else 0)

    pm.send('liveMapData', map_data_msg)
    _debug(f'Mapd *****: Publish: \n{map_data_msg}\n********')


# provides live map data information
def mapd_thread(sm=None, pm=None):
  mapd = MapD()
  rk = Ratekeeper(1., print_delay_threshold=None)  # Keeps rate at 1 hz

  # *** setup messaging
  if sm is None:
    sm = messaging.SubMaster(['gpsLocationExternal'])
  if pm is None:
    pm = messaging.PubMaster(['liveMapData'])

  while True:
    sm.update()
    mapd.update_gps(sm)
    mapd.updated_osm_data()
    mapd.update_route()
    mapd.publish(pm, sm)
    rk.keep_time()


def main(sm=None, pm=None):
  mapd_thread(sm, pm)


if __name__ == "__main__":
  main()
