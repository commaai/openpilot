#!/usr/bin/env python3
import json
import math
import os
import time
import shutil
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
from laika.ephemeris import Ephemeris, EphemerisType, convert_ublox_gps_ephem, convert_ublox_glonass_ephem, parse_qcom_ephem
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, correct_measurements, process_measurements, read_raw_ublox, read_raw_qcom
from laika.opt import calc_pos_fix, get_posfix_sympy_fun, calc_vel_fix, get_velfix_sympy_func
from selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from selfdrive.locationd.models.gnss_kf import GNSSKalman
from selfdrive.locationd.models.gnss_kf import States as GStates
from system.swaglog import cloudlog

MAX_TIME_GAP = 10
EPHEMERIS_CACHE = 'LaikadEphemerisV2'
DOWNLOADS_CACHE_FOLDER = "/tmp/comma_download_cache/"
CACHE_VERSION = 0.2
POS_FIX_RESIDUAL_THRESHOLD = 100.0


class Laikad:
  def __init__(self, valid_const=("GPS", "GLONASS"), auto_fetch_navs=True, auto_update=False,
               valid_ephem_types=(EphemerisType.NAV,),
               save_ephemeris=False, use_qcom=False):
    """
    valid_const: GNSS constellation which can be used
    auto_fetch_navs: If true fetch navs from internet when needed
    auto_update: If true download AstroDog will download all files needed. This can be ephemeris or correction data like ionosphere.
    valid_ephem_types: Valid ephemeris types to be used by AstroDog
    save_ephemeris: If true saves and loads nav and orbit ephemeris to cache.
    """
    self.astro_dog = AstroDog(valid_const=valid_const, auto_update=auto_update, valid_ephem_types=valid_ephem_types, clear_old_ephemeris=True, cache_dir=DOWNLOADS_CACHE_FOLDER)
    self.gnss_kf = GNSSKalman(GENERATED_DIR, cython=True, erratic_clock=use_qcom)

    self.auto_fetch_navs = auto_fetch_navs
    self.orbit_fetch_executor: Optional[ProcessPoolExecutor] = None
    self.orbit_fetch_future: Optional[Future] = None

    self.last_fetch_navs_t = None
    self.got_first_gnss_msg = False
    self.last_cached_t = None
    self.save_ephemeris = save_ephemeris
    self.load_cache()

    self.posfix_functions = {constellation: get_posfix_sympy_fun(constellation) for constellation in (ConstellationId.GPS, ConstellationId.GLONASS)}
    self.velfix_function = get_velfix_sympy_func()
    self.last_fix_pos = None
    self.last_fix_t = None
    self.gps_week = None
    self.use_qcom = use_qcom

  def load_cache(self):
    if not self.save_ephemeris:
      return

    cache = Params().get(EPHEMERIS_CACHE)
    if not cache:
      return

    try:
      cache = json.loads(cache, object_hook=deserialize_hook)
      if cache['version'] == CACHE_VERSION:
        self.astro_dog.add_navs(cache['navs'])
        self.last_fetch_navs_t = cache['last_fetch_navs_t']
      else:
        cache['navs'] = {}
    except json.decoder.JSONDecodeError:
      cloudlog.exception("Error parsing cache")
    timestamp = self.last_fetch_navs_t.as_datetime() if self.last_fetch_navs_t is not None else 'Nan'
    cloudlog.debug(
      f"Loaded navs ({sum([len(v) for v in cache['navs']])}) cache with timestamp: {timestamp}. Unique orbit and nav sats: {list(cache['navs'].keys())} " +
      f"With time range: {[f'{start.as_datetime()}, {end.as_datetime()}' for (start,end) in self.astro_dog.navs_fetched_times._ranges]}")

  def cache_ephemeris(self, t: GPSTime):
    if self.save_ephemeris and (self.last_cached_t is None or t - self.last_cached_t > SECS_IN_MIN):
      put_nonblocking(EPHEMERIS_CACHE, json.dumps(
        {'version': CACHE_VERSION, 'last_fetch_navs_t': self.last_fetch_navs_t, 'navs': self.astro_dog.navs},
        cls=CacheSerializer))
      cloudlog.debug("Cache saved")
      self.last_cached_t = t

  def get_lsq_fix(self, t, measurements):
    if self.last_fix_t is None or abs(self.last_fix_t - t) > 0:
      min_measurements = 5 if any(p.constellation_id == ConstellationId.GLONASS for p in measurements) else 4
      position_solution, pr_residuals, pos_std = calc_pos_fix(measurements, self.posfix_functions, min_measurements=min_measurements)
      if len(position_solution) < 3:
        return None
      position_estimate = position_solution[:3]

      position_std_residual = np.median(np.abs(pr_residuals))
      position_std = np.median(np.abs(pos_std))/10
      position_std = max(position_std_residual, position_std) * np.ones(3)

      velocity_solution, prr_residuals, vel_std = calc_vel_fix(measurements, position_estimate, self.velfix_function, min_measurements=min_measurements)
      if len(velocity_solution) < 3:
        return None
      velocity_estimate = velocity_solution[:3]

      velocity_std_residual = np.median(np.abs(prr_residuals))
      velocity_std = np.median(np.abs(vel_std))/10
      velocity_std = max(velocity_std, velocity_std_residual) * np.ones(3)

      return position_estimate, position_std, velocity_estimate, velocity_std

  def is_good_report(self, gnss_msg):
    if gnss_msg.which() == 'drMeasurementReport' and self.use_qcom:
      constellation_id = ConstellationId.from_qcom_source(gnss_msg.drMeasurementReport.source)
      # TODO support GLONASS
      return constellation_id in [ConstellationId.GPS, ConstellationId.SBAS]
    elif gnss_msg.which() == 'measurementReport' and not self.use_qcom:
      return True
    else:
      return False

  def read_report(self, gnss_msg):
    if self.use_qcom:
      report = gnss_msg.drMeasurementReport
      week = report.gpsWeek
      tow = report.gpsMilliseconds / 1000.0
      new_meas = read_raw_qcom(report)
    else:
      report = gnss_msg.measurementReport
      week = report.gpsWeek
      tow = report.rcvTow
      new_meas = read_raw_ublox(report)
    return week, tow, new_meas

  def is_ephemeris(self, gnss_msg):
    if self.use_qcom:
      return gnss_msg.which() == 'drSvPoly'
    else:
      return gnss_msg.which() in ('ephemeris', 'glonassEphemeris')

  def read_ephemeris(self, gnss_msg):
    if self.use_qcom:
      # TODO this is not robust to gps week rollover
      if self.gps_week is None:
        return
      ephem = parse_qcom_ephem(gnss_msg.drSvPoly, self.gps_week)
    else:
      if gnss_msg.which() == 'ephemeris':
        ephem = convert_ublox_gps_ephem(gnss_msg.ephemeris)
      elif gnss_msg.which() == 'glonassEphemeris':
        ephem = convert_ublox_glonass_ephem(gnss_msg.glonassEphemeris)
      else:
        cloudlog.error(f"Unsupported ephemeris type: {gnss_msg.which()}")
        return
    self.astro_dog.add_navs({ephem.prn: [ephem]})
    self.cache_ephemeris(t=ephem.epoch)

  def process_report(self, new_meas, t):
    # Filter measurements with unexpected pseudoranges for GPS and GLONASS satellites
    new_meas = [m for m in new_meas if 1e7 < m.observables['C1C'] < 3e7]
    processed_measurements = process_measurements(new_meas, self.astro_dog)
    if self.last_fix_pos is not None:
      corrected_measurements = correct_measurements(processed_measurements, self.last_fix_pos, self.astro_dog)
      instant_fix = self.get_lsq_fix(t, corrected_measurements)
      #instant_fix = self.get_lsq_fix(t, processed_measurements)
    else:
      corrected_measurements = []
      instant_fix = self.get_lsq_fix(t, processed_measurements)
    if instant_fix is None:
      return None
    else:
      position_estimate, position_std, velocity_estimate, velocity_std = instant_fix
      self.last_fix_t = t
      self.last_fix_pos = position_estimate
      self.lat_fix_pos_std = position_std
    if (t*1e9) % 10 == 0:
      cloudlog.debug(f"Measurements Incoming/Processed/Corrected: {len(new_meas), len(processed_measurements), len(corrected_measurements)}")
    return position_estimate, position_std, velocity_estimate, velocity_std, corrected_measurements, processed_measurements

  def process_gnss_msg(self, gnss_msg, gnss_mono_time: int, block=False):
    if self.is_ephemeris(gnss_msg):
      self.read_ephemeris(gnss_msg)
      return None
    elif self.is_good_report(gnss_msg):

      week, tow, new_meas = self.read_report(gnss_msg)
      self.gps_week = week
      if len(new_meas) == 0:
        return None

      t = gnss_mono_time * 1e-9
      if week > 0:
        self.got_first_gnss_msg = True
        latest_msg_t = GPSTime(week, tow)
        if self.auto_fetch_navs:
          self.fetch_navs(latest_msg_t, block)

      output = self.process_report(new_meas, t)
      if output is None:
        return None
      position_estimate, position_std, velocity_estimate, velocity_std, corrected_measurements, _ = output

      self.update_localizer(position_estimate, t, corrected_measurements)
      meas_msgs = [create_measurement_msg(m) for m in corrected_measurements]
      msg = messaging.new_message("gnssMeasurements")
      measurement_msg = log.LiveLocationKalman.Measurement.new_message

      P_diag = self.gnss_kf.P.diagonal()
      kf_valid = all(self.kf_valid(t))
      msg.gnssMeasurements = {
        "gpsWeek": week,
        "gpsTimeOfWeek": tow,
        "kalmanPositionECEF": measurement_msg(value=self.gnss_kf.x[GStates.ECEF_POS].tolist(),
                                        std=np.sqrt(P_diag[GStates.ECEF_POS]).tolist(),
                                        valid=kf_valid),
        "kalmanVelocityECEF": measurement_msg(value=self.gnss_kf.x[GStates.ECEF_VELOCITY].tolist(),
                                        std=np.sqrt(P_diag[GStates.ECEF_VELOCITY]).tolist(),
                                        valid=kf_valid),
        "positionECEF": measurement_msg(value=position_estimate, std=position_std.tolist(), valid=bool(self.last_fix_t == t)),
        "velocityECEF": measurement_msg(value=velocity_estimate, std=velocity_std.tolist(), valid=bool(self.last_fix_t == t)),

        "measTime": gnss_mono_time,
        "correctedMeasurements": meas_msgs
      }
      return msg

    #elif gnss_msg.which() == 'ionoData':
    # TODO: add this, Needed to better correct messages offline. First fix ublox_msg.cc to sent them.


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

  def fetch_navs(self, t: GPSTime, block):
    # Download new navs if 1 hour of navs data left
    if t + SECS_IN_HR not in self.astro_dog.navs_fetched_times and (self.last_fetch_navs_t is None or abs(t - self.last_fetch_navs_t) > SECS_IN_MIN):
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
          self.last_fetch_navs_t = ret[2]
        else:
          self.astro_dog.navs, self.astro_dog.navs_fetched_times, self.last_fetch_navs_t = ret
          self.cache_ephemeris(t=t)


def get_orbit_data(t: GPSTime, valid_const, auto_update, valid_ephem_types, cache_dir):
  astro_dog = AstroDog(valid_const=valid_const, auto_update=auto_update, valid_ephem_types=valid_ephem_types, cache_dir=cache_dir)
  cloudlog.info(f"Start to download/parse navs for time {t.as_datetime()}")
  start_time = time.monotonic()
  try:
    astro_dog.get_navs(t)
    cloudlog.info(f"Done parsing navs. Took {time.monotonic() - start_time:.1f}s")
    cloudlog.debug(f"Downloaded navs ({sum([len(v) for v in astro_dog.navs])}): {list(astro_dog.navs.keys())}" +
                   f"With time range: {[f'{start.as_datetime()}, {end.as_datetime()}' for (start,end) in astro_dog.orbit_fetched_times._ranges]}")
    return astro_dog.navs, astro_dog.navs_fetched_times, t
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
  week, time_of_week = -1, -1
  if ephem.eph_type == EphemerisType.NAV:
    source_type = EphemerisSourceType.nav
  elif ephem.eph_type == EphemerisType.QCOM_POLY:
    source_type = EphemerisSourceType.qcom
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
  qcom = 3


def process_msg(laikad, gnss_msg, mono_time, block=False):
  # TODO: Understand and use remaining unknown constellations
  if gnss_msg.which() == "drMeasurementReport":
    if getattr(gnss_msg, gnss_msg.which()).source not in ['glonass', 'gps', 'beidou', 'sbas']:
      return None

    if getattr(gnss_msg, gnss_msg.which()).gpsWeek > np.iinfo(np.int16).max:
      # gpsWeek 65535 is received rarely from quectel, this cannot be
      # passed to GnssMeasurements's gpsWeek (Int16)
      return None

  return laikad.process_gnss_msg(gnss_msg, mono_time, block=block)


def clear_tmp_cache():
  if os.path.exists(DOWNLOADS_CACHE_FOLDER):
    shutil.rmtree(DOWNLOADS_CACHE_FOLDER)
  os.mkdir(DOWNLOADS_CACHE_FOLDER)


def main(sm=None, pm=None, qc=None):
  #clear_tmp_cache()

  use_qcom = not Params().get_bool("UbloxAvailable", block=True)
  if use_qcom or (qc is not None and qc):
    raw_gnss_socket = "qcomGnss"
  else:
    raw_gnss_socket = "ubloxGnss"

  if sm is None:
    sm = messaging.SubMaster([raw_gnss_socket, 'clocks'])
  if pm is None:
    pm = messaging.PubMaster(['gnssMeasurements'])

  # disable until set as main gps source, to better analyze startup time
  use_internet = False #"LAIKAD_NO_INTERNET" not in os.environ

  replay = "REPLAY" in os.environ
  if replay or "CI" in os.environ:
    use_internet = True

  laikad = Laikad(save_ephemeris=not replay, auto_fetch_navs=use_internet, use_qcom=use_qcom)

  while True:
    sm.update()

    if sm.updated[raw_gnss_socket]:
      gnss_msg = sm[raw_gnss_socket]

      msg = process_msg(laikad, gnss_msg, sm.logMonoTime[raw_gnss_socket], replay)
      if msg is None:
        # TODO: beautify this, locationd needs a valid message
        msg = messaging.new_message("gnssMeasurements")
      pm.send('gnssMeasurements', msg)

    if not laikad.got_first_gnss_msg and sm.updated['clocks']:
      clocks_msg = sm['clocks']
      t = GPSTime.from_datetime(datetime.utcfromtimestamp(clocks_msg.wallTimeNanos * 1E-9))
      if laikad.auto_fetch_navs:
        laikad.fetch_navs(t, block=replay)

if __name__ == "__main__":
  main()
