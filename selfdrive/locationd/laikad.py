#!/usr/bin/env python3
import json
import time
from concurrent.futures import Future, ProcessPoolExecutor
from typing import List, Optional

import numpy as np
from collections import defaultdict

import sympy
from numpy.linalg import linalg

from cereal import log, messaging
from common.params import Params, put_nonblocking
from laika import AstroDog
from laika.constants import EARTH_ROTATION_RATE, SECS_IN_HR, SECS_IN_MIN, SPEED_OF_LIGHT
from laika.ephemeris import Ephemeris, EphemerisType, convert_ublox_ephem
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, correct_measurements, process_measurements, read_raw_ublox
from selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from selfdrive.locationd.models.gnss_kf import GNSSKalman
from selfdrive.locationd.models.gnss_kf import States as GStates
from system.swaglog import cloudlog

MAX_TIME_GAP = 10
EPHEMERIS_CACHE = 'LaikadEphemeris'
CACHE_VERSION = 0.1


class Laikad:
  def __init__(self, valid_const=("GPS", "GLONASS"), auto_update=False, valid_ephem_types=(EphemerisType.ULTRA_RAPID_ORBIT, EphemerisType.NAV),
               save_ephemeris=False):
    self.astro_dog = AstroDog(valid_const=valid_const, auto_update=auto_update, valid_ephem_types=valid_ephem_types, clear_old_ephemeris=True)
    self.gnss_kf = GNSSKalman(GENERATED_DIR)
    self.orbit_fetch_executor = ProcessPoolExecutor()
    self.orbit_fetch_future: Optional[Future] = None
    self.last_fetch_orbits_t = None
    self.last_cached_t = None
    self.save_ephemeris = save_ephemeris
    self.load_cache()
    self.posfix_functions = {constellation: get_posfix_sympy_fun(constellation) for constellation in (ConstellationId.GPS, ConstellationId.GLONASS)}

  def load_cache(self):
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

  def cache_ephemeris(self, t: GPSTime):
    if self.save_ephemeris and (self.last_cached_t is None or t - self.last_cached_t > SECS_IN_MIN):
      put_nonblocking(EPHEMERIS_CACHE, json.dumps(
        {'version': CACHE_VERSION, 'last_fetch_orbits_t': self.last_fetch_orbits_t, 'orbits': self.astro_dog.orbits, 'nav': self.astro_dog.nav},
        cls=CacheSerializer))
      self.last_cached_t = t

  def process_ublox_msg(self, ublox_msg, ublox_mono_time: int, block=False):
    if ublox_msg.which == 'measurementReport':
      report = ublox_msg.measurementReport
      if report.gpsWeek > 0:
        latest_msg_t = GPSTime(report.gpsWeek, report.rcvTow)
        self.fetch_orbits(latest_msg_t + SECS_IN_MIN, block)
      new_meas = read_raw_ublox(report)
      processed_measurements = process_measurements(new_meas, self.astro_dog)

      min_measurements = 5 if any(p.constellation_id == ConstellationId.GLONASS for p in processed_measurements) else 4
      pos_fix = calc_pos_fix_gauss_newton(processed_measurements, self.posfix_functions, min_measurements=min_measurements)

      t = ublox_mono_time * 1e-9
      kf_pos_std = None
      if all(self.kf_valid(t)):
        self.gnss_kf.predict(t)
        kf_pos_std = np.sqrt(abs(self.gnss_kf.P[GStates.ECEF_POS].diagonal()))
      # If localizer is valid use its position to correct measurements
      if kf_pos_std is not None and linalg.norm(kf_pos_std) < 100:
        est_pos = self.gnss_kf.x[GStates.ECEF_POS]
      elif len(pos_fix) > 0 and abs(np.array(pos_fix[1])).mean() < 1000:
        est_pos = pos_fix[0][:3]
      else:
        est_pos = None
      corrected_measurements = []
      if est_pos is not None:
        corrected_measurements = correct_measurements(processed_measurements, est_pos, self.astro_dog)

      self.update_localizer(est_pos, t, corrected_measurements)
      kf_valid = all(self.kf_valid(t))

      ecef_pos = self.gnss_kf.x[GStates.ECEF_POS].tolist()
      ecef_vel = self.gnss_kf.x[GStates.ECEF_VELOCITY].tolist()

      pos_std = np.sqrt(abs(self.gnss_kf.P[GStates.ECEF_POS].diagonal())).tolist()
      vel_std = np.sqrt(abs(self.gnss_kf.P[GStates.ECEF_VELOCITY].diagonal())).tolist()

      meas_msgs = [create_measurement_msg(m) for m in corrected_measurements]
      dat = messaging.new_message("gnssMeasurements")
      measurement_msg = log.LiveLocationKalman.Measurement.new_message
      dat.gnssMeasurements = {
        "positionECEF": measurement_msg(value=ecef_pos, std=pos_std, valid=kf_valid),
        "velocityECEF": measurement_msg(value=ecef_vel, std=vel_std, valid=kf_valid),
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
      if not valid[0]:
        cloudlog.info("Init gnss kalman filter")
      elif not valid[1]:
        cloudlog.error("Time gap of over 10s detected, gnss kalman reset")
      elif not valid[2]:
        cloudlog.error("Gnss kalman filter state is nan")
      else:
        cloudlog.error("Gnss kalman std too far")

      if est_pos is None:
        cloudlog.info("Position fix not available when resetting kalman filter")
        return
      self.init_gnss_localizer(est_pos.tolist())
    if len(measurements) > 0:
      kf_add_observations(self.gnss_kf, t, measurements)
    else:
      # Ensure gnss filter is updated even with no new measurements
      self.gnss_kf.predict(t)

  def kf_valid(self, t: float):
    filter_time = self.gnss_kf.filter.filter_time
    return [filter_time is not None,
            filter_time is not None and abs(t - filter_time) < MAX_TIME_GAP,
            all(np.isfinite(self.gnss_kf.x[GStates.ECEF_POS])),
            linalg.norm(self.gnss_kf.P[GStates.ECEF_POS]) < 1e5]

  def init_gnss_localizer(self, est_pos):
    x_initial, p_initial_diag = np.copy(GNSSKalman.x_initial), np.copy(np.diagonal(GNSSKalman.P_initial))
    x_initial[GStates.ECEF_POS] = est_pos
    p_initial_diag[GStates.ECEF_POS] = 1000 ** 2

    self.gnss_kf.init_state(x_initial, covs_diag=p_initial_diag)

  def fetch_orbits(self, t: GPSTime, block):
    if t not in self.astro_dog.orbit_fetched_times and (self.last_fetch_orbits_t is None or t - self.last_fetch_orbits_t > SECS_IN_HR):
      astro_dog_vars = self.astro_dog.valid_const, self.astro_dog.auto_update, self.astro_dog.valid_ephem_types
      if self.orbit_fetch_future is None:
        self.orbit_fetch_future = self.orbit_fetch_executor.submit(get_orbit_data, t, *astro_dog_vars)
        if block:
          self.orbit_fetch_future.result()
      if self.orbit_fetch_future.done():
        ret = self.orbit_fetch_future.result()
        self.last_fetch_orbits_t = t
        if ret:
          self.astro_dog.orbits, self.astro_dog.orbit_fetched_times = ret
          self.cache_ephemeris(t=t)
        self.orbit_fetch_future = None


def get_orbit_data(t: GPSTime, valid_const, auto_update, valid_ephem_types):
  astro_dog = AstroDog(valid_const=valid_const, auto_update=auto_update, valid_ephem_types=valid_ephem_types)
  cloudlog.info(f"Start to download/parse orbits for time {t.as_datetime()}")
  start_time = time.monotonic()
  data = None
  try:
    astro_dog.get_orbit_data(t, only_predictions=True)
    data = (astro_dog.orbits, astro_dog.orbit_fetched_times)
  except RuntimeError as e:
    cloudlog.info(f"No orbit data found. {e}")
  cloudlog.info(f"Done parsing orbits. Took {time.monotonic() - start_time:.1f}s")
  return data


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
  return c


def kf_add_observations(gnss_kf: GNSSKalman, t: float, measurements: List[GNSSMeasurement]):
  ekf_data = defaultdict(list)
  for m in measurements:
    m_arr = m.as_array()
    if m.constellation_id == ConstellationId.GPS:
      ekf_data[ObservationKind.PSEUDORANGE_GPS].append(m_arr)
      ekf_data[ObservationKind.PSEUDORANGE_RATE_GPS].append(m_arr)
    elif m.constellation_id == ConstellationId.GLONASS:
      ekf_data[ObservationKind.PSEUDORANGE_GLONASS].append(m_arr)
      ekf_data[ObservationKind.PSEUDORANGE_RATE_GLONASS].append(m_arr)

  for kind, data in ekf_data.items():
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


def calc_pos_fix_gauss_newton(measurements, posfix_functions, x0=None, signal='C1C', min_measurements=6):
  '''
  Calculates gps fix using gauss newton method
  To solve the problem a minimal of 4 measurements are required.
    If Glonass is included 5 are required to solve for the additional free variable.
  returns:
  0 -> list with positions
  '''
  if x0 is None:
    x0 = [0, 0, 0, 0, 0]
  n = len(measurements)
  if n < min_measurements:
    return []

  Fx_pos = pr_residual(measurements, posfix_functions, signal=signal)
  x = gauss_newton(Fx_pos, x0)
  residual, _ = Fx_pos(x, weight=1.0)
  return x, residual


def pr_residual(measurements, posfix_functions, signal='C1C'):
  def Fx_pos(inp, weight=None):
    vals, gradients = [], []

    for meas in measurements:
      pr = meas.observables[signal]
      pr += meas.sat_clock_err * SPEED_OF_LIGHT

      w = (1 / meas.observables_std[signal]) if weight is None else weight

      val, *gradient = posfix_functions[meas.constellation_id](*inp, pr, *meas.sat_pos, w)
      vals.append(val)
      gradients.append(gradient)
    return np.asarray(vals), np.asarray(gradients)

  return Fx_pos


def gauss_newton(fun, b, xtol=1e-8, max_n=25):
  for _ in range(max_n):
    # Compute function and jacobian on current estimate
    r, J = fun(b)

    # Update estimate
    delta = np.linalg.pinv(J) @ r
    b -= delta

    # Check step size for stopping condition
    if np.linalg.norm(delta) < xtol:
      break
  return b


def get_posfix_sympy_fun(constellation):
  # Unknowns
  x, y, z = sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')
  bc = sympy.Symbol('bc')
  bg = sympy.Symbol('bg')
  var = [x, y, z, bc, bg]

  # Knowns
  pr = sympy.Symbol('pr')
  sat_x, sat_y, sat_z = sympy.Symbol('sat_x'), sympy.Symbol('sat_y'), sympy.Symbol('sat_z')
  weight = sympy.Symbol('weight')

  theta = EARTH_ROTATION_RATE * (pr - bc) / SPEED_OF_LIGHT
  val = sympy.sqrt(
    (sat_x * sympy.cos(theta) + sat_y * sympy.sin(theta) - x) ** 2 +
    (sat_y * sympy.cos(theta) - sat_x * sympy.sin(theta) - y) ** 2 +
    (sat_z - z) ** 2
  )

  if constellation == ConstellationId.GLONASS:
    res = weight * (val - (pr - bc - bg))
  elif constellation == ConstellationId.GPS:
    res = weight * (val - (pr - bc))
  else:
    raise NotImplementedError(f"Constellation {constellation} not supported")

  res = [res] + [sympy.diff(res, v) for v in var]

  return sympy.lambdify([x, y, z, bc, bg, pr, sat_x, sat_y, sat_z, weight], res)


def main():
  sm = messaging.SubMaster(['ubloxGnss'])
  pm = messaging.PubMaster(['gnssMeasurements'])

  laikad = Laikad(save_ephemeris=True)
  while True:
    sm.update()

    if sm.updated['ubloxGnss']:
      ublox_msg = sm['ubloxGnss']
      msg = laikad.process_ublox_msg(ublox_msg, sm.logMonoTime['ubloxGnss'])
      if msg is not None:
        pm.send('gnssMeasurements', msg)


if __name__ == "__main__":
  main()
