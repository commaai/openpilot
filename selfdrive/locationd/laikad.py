#!/usr/bin/env python3
import math
import os
import time
import shutil
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from enum import IntEnum
from typing import List, Optional, Dict, Any

import numpy as np

from cereal import log, messaging
from common.params import Params, put_nonblocking
from laika import AstroDog
from laika.constants import SECS_IN_HR, SECS_IN_MIN
from laika.downloader import DownloadFailed
from laika.ephemeris import EphemerisType, GPSEphemeris, GLONASSEphemeris, ephemeris_structs, parse_qcom_ephem
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId, get_sv_id
from laika.raw_gnss import GNSSMeasurement, correct_measurements, process_measurements, read_raw_ublox, read_raw_qcom
from laika.opt import calc_pos_fix, get_posfix_sympy_fun, calc_vel_fix, get_velfix_sympy_func
from selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from selfdrive.locationd.models.gnss_kf import GNSSKalman
from selfdrive.locationd.models.gnss_kf import States as GStates
from system.swaglog import cloudlog

MAX_TIME_GAP = 10
EPHEMERIS_CACHE = 'LaikadEphemerisV3'
DOWNLOADS_CACHE_FOLDER = "/tmp/comma_download_cache/"
CACHE_VERSION = 0.2
POS_FIX_RESIDUAL_THRESHOLD = 100.0


class LogEphemerisType(IntEnum):
  nav = 0
  nasaUltraRapid = 1
  glonassIacUltraRapid = 2
  qcom = 3

class EphemerisSource(IntEnum):
  gnssChip = 0
  internet = 1
  cache = 2
  unknown = 3

def get_log_eph_type(ephem):
  if ephem.eph_type == EphemerisType.NAV:
    source_type = LogEphemerisType.nav
  elif ephem.eph_type == EphemerisType.QCOM_POLY:
    source_type = LogEphemerisType.qcom
  else:
    assert ephem.file_epoch is not None
    file_src = ephem.file_source
    if file_src == 'igu':  # example nasa: '2214/igu22144_00.sp3.Z'
      source_type = LogEphemerisType.nasaUltraRapid
    elif file_src == 'Sta':  # example nasa: '22166/ultra/Stark_1D_22061518.sp3'
      source_type = LogEphemerisType.glonassIacUltraRapid
    else:
      raise Exception(f"Didn't expect file source {file_src}")
  return source_type

def get_log_eph_source(ephem):
  if ephem.file_name == 'qcom' or ephem.file_name == 'ublox':
    source = EphemerisSource.gnssChip
  elif ephem.file_name == EPHEMERIS_CACHE:
    source = EphemerisSource.cache
  else:
    source = EphemerisSource.internet
  return source


class Laikad:
  def __init__(self, valid_const=(ConstellationId.GPS, ConstellationId.GLONASS), auto_fetch_navs=True, auto_update=False,
               valid_ephem_types=(EphemerisType.NAV, EphemerisType.QCOM_POLY),
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
    self.got_first_gnss_msg = False

    self.last_report_time = GPSTime(0, 0)
    self.last_fetch_navs_t = GPSTime(0, 0)
    self.last_cached_t = GPSTime(0, 0)
    self.save_ephemeris = save_ephemeris
    self.load_cache()

    self.posfix_functions = {constellation: get_posfix_sympy_fun(constellation) for constellation in (ConstellationId.GPS, ConstellationId.GLONASS)}
    self.velfix_function = get_velfix_sympy_func()
    self.last_fix_pos = None
    self.last_fix_t = None
    self.gps_week = None
    self.use_qcom = use_qcom
    self.first_log_time = None
    self.ttff = -1

  def load_cache(self):
    if not self.save_ephemeris:
      return

    cache_bytes = Params().get(EPHEMERIS_CACHE)
    if not cache_bytes:
      return

    nav_dict = {}
    try:
      ephem_cache = ephemeris_structs.EphemerisCache.from_bytes(cache_bytes)
      glonass_navs = [GLONASSEphemeris(data_struct, file_name=EPHEMERIS_CACHE) for data_struct in ephem_cache.glonassEphemerides]
      gps_navs = [GPSEphemeris(data_struct, file_name=EPHEMERIS_CACHE) for data_struct in ephem_cache.gpsEphemerides]
      for e in sum([glonass_navs, gps_navs], []):
        if e.prn not in nav_dict:
          nav_dict[e.prn] = []
        nav_dict[e.prn].append(e)
      self.astro_dog.add_navs(nav_dict)
    except Exception:
      cloudlog.exception("Error parsing cache")
    cloudlog.debug(
      f"Loaded navs ({sum([len(nav_dict[prn]) for prn in nav_dict.keys()])}). Unique orbit and nav sats: {list(nav_dict.keys())} ")

  def cache_ephemeris(self):

    if self.save_ephemeris and (self.last_report_time - self.last_cached_t > SECS_IN_MIN):
      nav_list: List = sum([v for k,v in self.astro_dog.navs.items()], [])
      ephem_cache = ephemeris_structs.EphemerisCache(**{'glonassEphemerides': [e.data for e in nav_list if e.prn[0]=='R'],
                                                        'gpsEphemerides': [e.data for e in nav_list if e.prn[0]=='G']})

      put_nonblocking(EPHEMERIS_CACHE, ephem_cache.to_bytes())
      cloudlog.debug("Cache saved")
      self.last_cached_t = self.last_report_time

  def create_ephem_statuses(self):
    ephemeris_statuses = []
    prns_to_check = list(self.astro_dog.get_all_ephem_prns())
    prns_to_check.sort()
    for prn in prns_to_check:
      eph = self.astro_dog.get_eph(prn, self.last_report_time)
      if eph is not None:
        status = log.GnssMeasurements.EphemerisStatus.new_message()
        status.constellationId = ConstellationId.from_rinex_char(prn[0]).value
        status.svId = get_sv_id(prn)
        status.type = get_log_eph_type(eph).value
        status.source = get_log_eph_source(eph).value
        ephemeris_statuses.append(status)
    return ephemeris_statuses


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
      # TODO: Understand and use remaining unknown constellations
      try:
        good_constellation = constellation_id in [ConstellationId.GPS, ConstellationId.SBAS]
      except NotImplementedError:
        good_constellation = False
      # gpsWeek 65535 is received rarely from quectel, this cannot be
      # passed to GnssMeasurements's gpsWeek (Int16)
      good_week = not getattr(gnss_msg, gnss_msg.which()).gpsWeek > np.iinfo(np.int16).max
      return good_constellation and good_week
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
    self.last_report_time = GPSTime(week, tow)
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
      try:
        ephem = parse_qcom_ephem(gnss_msg.drSvPoly, self.gps_week)
        self.astro_dog.add_qcom_polys({ephem.prn: [ephem]})
      except Exception:
        cloudlog.exception("Error parsing qcom svPoly ephemeris from qcom module")
        return

    else:
      if gnss_msg.which() == 'ephemeris':
        data_struct = ephemeris_structs.Ephemeris.new_message(**gnss_msg.ephemeris.to_dict())
        try:
          ephem = GPSEphemeris(data_struct, file_name='ublox')
        except Exception:
          cloudlog.exception("Error parsing GPS ephemeris from ublox")
          return
      elif gnss_msg.which() == 'glonassEphemeris':
        data_struct = ephemeris_structs.GlonassEphemeris.new_message(**gnss_msg.glonassEphemeris.to_dict())
        try:
          ephem = GLONASSEphemeris(data_struct, file_name='ublox')
        except Exception:
          cloudlog.exception("Error parsing GLONASS ephemeris from ublox")
          return
      else:
        cloudlog.error(f"Unsupported ephemeris type: {gnss_msg.which()}")
        return
      self.astro_dog.add_navs({ephem.prn: [ephem]})
    self.cache_ephemeris()

  def process_report(self, new_meas, t):
    # Filter measurements with unexpected pseudoranges for GPS and GLONASS satellites
    new_meas = [m for m in new_meas if 1e7 < m.observables['C1C'] < 3e7]
    processed_measurements = process_measurements(new_meas, self.astro_dog)
    if self.last_fix_pos is not None:
      est_pos = self.last_fix_pos
    else:
      est_pos = self.gnss_kf.x[GStates.ECEF_POS].tolist()
    corrected_measurements = correct_measurements(processed_measurements, est_pos, self.astro_dog)
    return corrected_measurements

  def calc_fix(self, t, measurements):
    instant_fix = self.get_lsq_fix(t, measurements)
    if instant_fix is None:
      return None
    else:
      position_estimate, position_std, velocity_estimate, velocity_std = instant_fix
      self.last_fix_t = t
      self.last_fix_pos = position_estimate
      self.lat_fix_pos_std = position_std
      return position_estimate, position_std, velocity_estimate, velocity_std

  def process_gnss_msg(self, gnss_msg, gnss_mono_time: int, block=False):
    out_msg = messaging.new_message("gnssMeasurements")
    t = gnss_mono_time * 1e-9
    msg_dict: Dict[str, Any] = {"measTime": gnss_mono_time}
    if self.first_log_time is None:
      self.first_log_time = 1e-9 * gnss_mono_time
    if self.is_ephemeris(gnss_msg):
      self.read_ephemeris(gnss_msg)
    elif self.is_good_report(gnss_msg):
      week, tow, new_meas = self.read_report(gnss_msg)
      self.gps_week = week
      if week > 0:
        self.got_first_gnss_msg = True
        latest_msg_t = GPSTime(week, tow)
        if self.auto_fetch_navs:
          self.fetch_navs(latest_msg_t, block)

      corrected_measurements = self.process_report(new_meas, t)
      msg_dict['correctedMeasurements'] = [create_measurement_msg(m) for m in corrected_measurements]

      fix = self.calc_fix(t, corrected_measurements)
      measurement_msg = log.LiveLocationKalman.Measurement.new_message
      if fix is not None:
        position_estimate, position_std, velocity_estimate, velocity_std = fix
        if self.ttff <= 0:
          self.ttff = max(1e-3, t - self.first_log_time)
        msg_dict["positionECEF"] = measurement_msg(value=position_estimate, std=position_std.tolist(), valid=bool(self.last_fix_t == t))
        msg_dict["velocityECEF"] = measurement_msg(value=velocity_estimate, std=velocity_std.tolist(), valid=bool(self.last_fix_t == t))

      self.update_localizer(self.last_fix_pos, t, corrected_measurements)
      P_diag = self.gnss_kf.P.diagonal()
      kf_valid = all(self.kf_valid(t))
      msg_dict["kalmanPositionECEF"] = measurement_msg(value=self.gnss_kf.x[GStates.ECEF_POS].tolist(),
                                        std=np.sqrt(P_diag[GStates.ECEF_POS]).tolist(),
                                        valid=kf_valid)
      msg_dict["kalmanVelocityECEF"] = measurement_msg(value=self.gnss_kf.x[GStates.ECEF_VELOCITY].tolist(),
                                        std=np.sqrt(P_diag[GStates.ECEF_VELOCITY]).tolist(),
                                        valid=kf_valid)

    msg_dict['gpsWeek'] = self.last_report_time.week
    msg_dict['gpsTimeOfWeek'] = self.last_report_time.tow
    msg_dict['timeToFirstFix'] = self.ttff
    msg_dict['ephemerisStatuses'] = self.create_ephem_statuses()
    out_msg.gnssMeasurements = msg_dict
    return out_msg

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
      if est_pos is not None and len(est_pos) > 0:
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
    if t + SECS_IN_HR not in self.astro_dog.navs_fetched_times and (abs(t - self.last_fetch_navs_t) > SECS_IN_MIN):
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
          self.cache_ephemeris()


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


def clear_tmp_cache():
  if os.path.exists(DOWNLOADS_CACHE_FOLDER):
    shutil.rmtree(DOWNLOADS_CACHE_FOLDER)
  os.mkdir(DOWNLOADS_CACHE_FOLDER)


def main(sm=None, pm=None):
  #clear_tmp_cache()

  use_qcom = not Params().get_bool("UbloxAvailable", block=True)
  if use_qcom:
    raw_name = "qcomGnss"
  else:
    raw_name = "ubloxGnss"
  raw_gnss_sock = messaging.sub_sock(raw_name, conflate=False, timeout=1000)

  if sm is None:
    sm = messaging.SubMaster(['clocks',])
  if pm is None:
    pm = messaging.PubMaster(['gnssMeasurements'])

  # disable until set as main gps source, to better analyze startup time
  use_internet = False  # "LAIKAD_NO_INTERNET" not in os.environ

  replay = "REPLAY" in os.environ
  if replay or "CI" in os.environ:
    use_internet = True

  laikad = Laikad(save_ephemeris=not replay, auto_fetch_navs=use_internet, use_qcom=use_qcom)

  while True:
    for in_msg in messaging.drain_sock(raw_gnss_sock):
      out_msg = laikad.process_gnss_msg(getattr(in_msg, raw_name), in_msg.logMonoTime, replay)
      pm.send('gnssMeasurements', out_msg)

    sm.update(0)
    if not laikad.got_first_gnss_msg and sm.updated['clocks']:
      clocks_msg = sm['clocks']
      t = GPSTime.from_datetime(datetime.utcfromtimestamp(clocks_msg.wallTimeNanos * 1E-9))
      if laikad.auto_fetch_navs:
        laikad.fetch_navs(t, block=replay)


if __name__ == "__main__":
  main()
