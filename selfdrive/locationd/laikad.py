#!/usr/bin/env python3
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from enum import IntEnum
from typing import List, Optional

import numpy as np

from cereal import log, messaging
from common.params import Params, put_nonblocking
from laika import AstroDog
from laika.constants import SECS_IN_HR, SECS_IN_MIN
from laika.downloader import DownloadFailed
from laika.ephemeris import Ephemeris, EphemerisType, convert_ublox_ephem
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, correct_measurements, process_measurements, read_raw_ublox
from selfdrive.locationd.laikad_helpers import calc_pos_fix_gauss_newton, get_posfix_sympy_fun
from selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from selfdrive.locationd.models.gnss_kf import GNSSKalman
from selfdrive.locationd.models.gnss_kf import States as GStates
from system.swaglog import cloudlog

MAX_TIME_GAP = 10
EPHEMERIS_CACHE = 'LaikadEphemeris'
DOWNLOADS_CACHE_FOLDER = "/tmp/comma_download_cache"
CACHE_VERSION = 0.1
POS_FIX_RESIDUAL_THRESHOLD = 100.0


class Laikad:
  def __init__(self, valid_const=("GPS", "GLONASS"), auto_fetch_orbits=True, auto_update=False,
               valid_ephem_types=(EphemerisType.ULTRA_RAPID_ORBIT, EphemerisType.NAV),
               save_ephemeris=False):
    """
    valid_const: GNSS constellation which can be used
    auto_fetch_orbits: If true fetch orbits from internet when needed
    auto_update: If true download AstroDog will download all files needed. This can be ephemeris or correction data like ionosphere.
    valid_ephem_types: Valid ephemeris types to be used by AstroDog
    save_ephemeris: If true saves and loads nav and orbit ephemeris to cache.
    """
    self.astro_dog = AstroDog(valid_const=valid_const, auto_update=auto_update, valid_ephem_types=valid_ephem_types, clear_old_ephemeris=True, cache_dir=DOWNLOADS_CACHE_FOLDER)
    self.gnss_kf = GNSSKalman(GENERATED_DIR, cython=True)

    self.auto_fetch_orbits = auto_fetch_orbits
    self.orbit_fetch_executor: Optional[ProcessPoolExecutor] = None
    self.orbit_fetch_future: Optional[Future] = None

    self.last_fetch_orbits_t = None
    self.got_first_ublox_msg = False
    self.last_cached_t = None
    self.save_ephemeris = save_ephemeris
    self.load_cache()

    self.posfix_functions = {constellation: get_posfix_sympy_fun(constellation) for constellation in (ConstellationId.GPS, ConstellationId.GLONASS)}
    self.last_pos_fix = []
    self.last_pos_residual = []
    self.last_pos_fix_t = None

  def load_cache(self):
    if not self.save_ephemeris:
      return

    cache = Params().get(EPHEMERIS_CACHE)
    if not cache:
      return

    try:
      cache = json.loads(cache, object_hook=deserialize_hook)
      self.astro_dog.add_orbits(cache['orbits'])
      self.astro_dog.add_navs(cache['nav'])
      self.last_fetch_orbits_t = cache['last_fetch_orbits_t']
    except json.decoder.JSONDecodeError:
      cloudlog.exception("Error parsing cache")
    timestamp = self.last_fetch_orbits_t.as_datetime() if self.last_fetch_orbits_t is not None else 'Nan'
    cloudlog.debug(
      f"Loaded nav ({sum([len(v) for v in cache['nav']])}) and orbits ({sum([len(v) for v in cache['orbits']])}) cache with timestamp: {timestamp}. Unique orbit and nav sats: {list(cache['orbits'].keys())} {list(cache['nav'].keys())} " +
      f"With time range: {[f'{start.as_datetime()}, {end.as_datetime()}' for (start,end) in self.astro_dog.orbit_fetched_times._ranges]}")

  def cache_ephemeris(self, t: GPSTime):
    if self.save_ephemeris and (self.last_cached_t is None or t - self.last_cached_t > SECS_IN_MIN):
      put_nonblocking(EPHEMERIS_CACHE, json.dumps(
        {'version': CACHE_VERSION, 'last_fetch_orbits_t': self.last_fetch_orbits_t, 'orbits': self.astro_dog.orbits, 'nav': self.astro_dog.nav},
        cls=CacheSerializer))
      cloudlog.debug("Cache saved")
      self.last_cached_t = t

  def get_est_pos(self, t, processed_measurements):
    if self.last_pos_fix_t is None or abs(self.last_pos_fix_t - t) >= 2:
      min_measurements = 6 if any(p.constellation_id == ConstellationId.GLONASS for p in processed_measurements) else 5
      pos_fix, pos_fix_residual = calc_pos_fix_gauss_newton(processed_measurements, self.posfix_functions, min_measurements=min_measurements)
      if len(pos_fix) > 0:
        self.last_pos_fix_t = t
        residual_median = np.median(np.abs(pos_fix_residual))
        if np.median(np.abs(pos_fix_residual)) < POS_FIX_RESIDUAL_THRESHOLD:
          cloudlog.debug(f"Pos fix is within threshold with median: {residual_median.round()}")
          self.last_pos_fix = pos_fix[:3]
          self.last_pos_residual = pos_fix_residual
        else:
          cloudlog.debug(f"Pos fix failed with median: {residual_median.round()}. All residuals: {np.round(pos_fix_residual)}")
    return self.last_pos_fix

  def process_ublox_msg(self, ublox_msg, ublox_mono_time: int, block=False):
    if ublox_msg.which == 'measurementReport':
      t = ublox_mono_time * 1e-9
      report = ublox_msg.measurementReport
      if report.gpsWeek > 0:
        self.got_first_ublox_msg = True
        latest_msg_t = GPSTime(report.gpsWeek, report.rcvTow)
        if self.auto_fetch_orbits:
          self.fetch_orbits(latest_msg_t, block)

      new_meas = read_raw_ublox(report)
      # Filter measurements with unexpected pseudoranges for GPS and GLONASS satellites
      new_meas = [m for m in new_meas if 1e7 < m.observables['C1C'] < 3e7]

      processed_measurements = process_measurements(new_meas, self.astro_dog)
      est_pos = self.get_est_pos(t, processed_measurements)

      corrected_measurements = correct_measurements(processed_measurements, est_pos, self.astro_dog) if len(est_pos) > 0 else []
      if ublox_mono_time % 10 == 0:
        cloudlog.debug(f"Measurements Incoming/Processed/Corrected: {len(new_meas), len(processed_measurements), len(corrected_measurements)}")

      self.update_localizer(est_pos, t, corrected_measurements)
      kf_valid = all(self.kf_valid(t))
      ecef_pos = self.gnss_kf.x[GStates.ECEF_POS]
      ecef_vel = self.gnss_kf.x[GStates.ECEF_VELOCITY]

      p = self.gnss_kf.P.diagonal()
      pos_std = np.sqrt(p[GStates.ECEF_POS])
      vel_std = np.sqrt(p[GStates.ECEF_VELOCITY])

      meas_msgs = [create_measurement_msg(m) for m in corrected_measurements]
      dat = messaging.new_message("gnssMeasurements")
      measurement_msg = log.LiveLocationKalman.Measurement.new_message
      dat.gnssMeasurements = {
        "gpsWeek": report.gpsWeek,
        "gpsTimeOfWeek": report.rcvTow,
        "positionECEF": measurement_msg(value=ecef_pos.tolist(), std=pos_std.tolist(), valid=kf_valid),
        "velocityECEF": measurement_msg(value=ecef_vel.tolist(), std=vel_std.tolist(), valid=kf_valid),
        "positionFixECEF": measurement_msg(value=self.last_pos_fix, std=self.last_pos_residual, valid=self.last_pos_fix_t == t),
        "ubloxMonoTime": ublox_mono_time,
        "correctedMeasurements": meas_msgs
      }
      return dat
    elif ublox_msg.which == 'ephemeris':
      ephem = convert_ublox_ephem(ublox_msg.ephemeris)
      self.astro_dog.add_navs({ephem.prn: [ephem]})
      self.cache_ephemeris(t=ephem.epoch)
    # elif ublox_msg.which == 'ionoData':
    # todo add this. Needed to better correct messages offline. First fix ublox_msg.cc to sent them.

  def update_localizer(self, est_pos, t: float, measurements: List[GNSSMeasurement]):
    # Check time and outputs are valid
    valid = self.kf_valid(t)
    if not all(valid):
      if not valid[0]:  # Filter not initialized
        pass
      elif not valid[1]:
        cloudlog.error("Time gap of over 10s detected, gnss kalman reset")
      elif not valid[2]:
        cloudlog.error("Gnss kalman filter state is nan")
      if len(est_pos) > 0:
        cloudlog.info(f"Reset kalman filter with {est_pos}")
        self.init_gnss_localizer(est_pos)
      else:
        return
    if len(measurements) > 0:
      kf_add_observations(self.gnss_kf, t, measurements)
    else:
      # Ensure gnss filter is updated even with no new measurements
      self.gnss_kf.predict(t)

  def kf_valid(self, t: float) -> List[bool]:
    filter_time = self.gnss_kf.filter.get_filter_time()
    return [not math.isnan(filter_time),
            abs(t - filter_time) < MAX_TIME_GAP,
            all(np.isfinite(self.gnss_kf.x[GStates.ECEF_POS]))]

  def init_gnss_localizer(self, est_pos):
    x_initial, p_initial_diag = np.copy(GNSSKalman.x_initial), np.copy(np.diagonal(GNSSKalman.P_initial))
    x_initial[GStates.ECEF_POS] = est_pos
    p_initial_diag[GStates.ECEF_POS] = 1000 ** 2
    self.gnss_kf.init_state(x_initial, covs_diag=p_initial_diag)

  def fetch_orbits(self, t: GPSTime, block):
    # Download new orbits if 1 hour of orbits data left
    if t + SECS_IN_HR not in self.astro_dog.orbit_fetched_times and (self.last_fetch_orbits_t is None or abs(t - self.last_fetch_orbits_t) > SECS_IN_MIN):
      astro_dog_vars = self.astro_dog.valid_const, self.astro_dog.auto_update, self.astro_dog.valid_ephem_types, self.astro_dog.cache_dir
      ret = None

      if block:  # Used for testing purposes
        ret = get_orbit_data(t, *astro_dog_vars)
      elif self.orbit_fetch_future is None:
        self.orbit_fetch_executor = ProcessPoolExecutor(max_workers=1)
        self.orbit_fetch_future = self.orbit_fetch_executor.submit(get_orbit_data, t, *astro_dog_vars)
      elif self.orbit_fetch_future.done():
        ret = self.orbit_fetch_future.result()
        self.orbit_fetch_executor = self.orbit_fetch_future = None

      if ret is not None:
        if ret[0] is None:
          self.last_fetch_orbits_t = ret[2]
        else:
          self.astro_dog.orbits, self.astro_dog.orbit_fetched_times, self.last_fetch_orbits_t = ret
          self.cache_ephemeris(t=t)


def get_orbit_data(t: GPSTime, valid_const, auto_update, valid_ephem_types, cache_dir):
  astro_dog = AstroDog(valid_const=valid_const, auto_update=auto_update, valid_ephem_types=valid_ephem_types, cache_dir=cache_dir)
  cloudlog.info(f"Start to download/parse orbits for time {t.as_datetime()}")
  start_time = time.monotonic()
  try:
    astro_dog.get_orbit_data(t, only_predictions=True)
    cloudlog.info(f"Done parsing orbits. Took {time.monotonic() - start_time:.1f}s")
    cloudlog.debug(f"Downloaded orbits ({sum([len(v) for v in astro_dog.orbits])}): {list(astro_dog.orbits.keys())}" +
                   f"With time range: {[f'{start.as_datetime()}, {end.as_datetime()}' for (start,end) in astro_dog.orbit_fetched_times._ranges]}")
    return astro_dog.orbits, astro_dog.orbit_fetched_times, t
  except (DownloadFailed, RuntimeError, ValueError, IOError) as e:
    cloudlog.warning(f"No orbit data found or parsing failure: {e}")
  return None, None, t


def create_measurement_msg(meas: GNSSMeasurement):
  c = log.GnssMeasurements.CorrectedMeasurement.new_message()
  c.constellationId = meas.constellation_id.value
  c.svId = meas.sv_id
  c.glonassFrequency = meas.glonass_freq if meas.constellation_id == ConstellationId.GLONASS else 0
  c.pseudorange = float(meas.observables_final['C1C'])
  c.pseudorangeStd = float(meas.observables_std['C1C'])
  c.pseudorangeRate = float(meas.observables_final['D1C'])
  c.pseudorangeRateStd = float(meas.observables_std['D1C'])
  c.satPos = meas.sat_pos_final.tolist()
  c.satVel = meas.sat_vel.tolist()
  c.satVel = meas.sat_vel.tolist()
  ephem = meas.sat_ephemeris
  assert ephem is not None
  if ephem.eph_type == EphemerisType.NAV:
    source_type = EphemerisSourceType.nav
    week, time_of_week = -1, -1
  else:
    assert ephem.file_epoch is not None
    week = ephem.file_epoch.week
    time_of_week = ephem.file_epoch.tow
    file_src = ephem.file_source
    if file_src == 'igu':  # example nasa: '2214/igu22144_00.sp3.Z'
      source_type = EphemerisSourceType.nasaUltraRapid
    elif file_src == 'Sta':  # example nasa: '22166/ultra/Stark_1D_22061518.sp3'
      source_type = EphemerisSourceType.glonassIacUltraRapid
    else:
      raise Exception(f"Didn't expect file source {file_src}")

  c.ephemerisSource.type = source_type.value
  c.ephemerisSource.gpsWeek = week
  c.ephemerisSource.gpsTimeOfWeek = int(time_of_week)
  return c


def kf_add_observations(gnss_kf: GNSSKalman, t: float, measurements: List[GNSSMeasurement]):
  ekf_data = defaultdict(list)
  for m in measurements:
    m_arr = m.as_array()
    if m.constellation_id == ConstellationId.GPS:
      ekf_data[ObservationKind.PSEUDORANGE_GPS].append(m_arr)
    elif m.constellation_id == ConstellationId.GLONASS:
      ekf_data[ObservationKind.PSEUDORANGE_GLONASS].append(m_arr)
  ekf_data[ObservationKind.PSEUDORANGE_RATE_GPS] = ekf_data[ObservationKind.PSEUDORANGE_GPS]
  ekf_data[ObservationKind.PSEUDORANGE_RATE_GLONASS] = ekf_data[ObservationKind.PSEUDORANGE_GLONASS]
  for kind, data in ekf_data.items():
    if len(data) > 0:
      gnss_kf.predict_and_observe(t, kind, data)


class CacheSerializer(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, Ephemeris):
      return o.to_json()
    if isinstance(o, GPSTime):
      return o.__dict__
    if isinstance(o, np.ndarray):
      return o.tolist()
    return json.JSONEncoder.default(self, o)


def deserialize_hook(dct):
  if 'ephemeris' in dct:
    return Ephemeris.from_json(dct)
  if 'week' in dct:
    return GPSTime(dct['week'], dct['tow'])
  return dct


class EphemerisSourceType(IntEnum):
  nav = 0
  nasaUltraRapid = 1
  glonassIacUltraRapid = 2


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['ubloxGnss', 'clocks'])
  if pm is None:
    pm = messaging.PubMaster(['gnssMeasurements'])

  replay = "REPLAY" in os.environ
  use_internet = "LAIKAD_NO_INTERNET" not in os.environ
  laikad = Laikad(save_ephemeris=not replay, auto_fetch_orbits=use_internet)

  while True:
    sm.update()

    if sm.updated['ubloxGnss']:
      ublox_msg = sm['ubloxGnss']
      msg = laikad.process_ublox_msg(ublox_msg, sm.logMonoTime['ubloxGnss'], block=replay)
      if msg is not None:
        pm.send('gnssMeasurements', msg)
    if not laikad.got_first_ublox_msg and sm.updated['clocks']:
      clocks_msg = sm['clocks']
      t = GPSTime.from_datetime(datetime.utcfromtimestamp(clocks_msg.wallTimeNanos * 1E-9))
      if laikad.auto_fetch_orbits:
        laikad.fetch_orbits(t, block=replay)


if __name__ == "__main__":
  main()
