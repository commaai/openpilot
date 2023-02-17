import json
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
import numpy.polynomial.polynomial as poly
from datetime import datetime, timedelta
from math import sin, cos, sqrt, fabs, atan2

from .gps_time import GPSTime, utc_to_gpst
from .constants import SPEED_OF_LIGHT, SECS_IN_MIN, SECS_IN_HR, SECS_IN_DAY, \
                       SECS_IN_WEEK, EARTH_ROTATION_RATE, EARTH_GM
from .helpers import get_constellation, get_prn_from_nmea_id


def read4(f, rinex_ver):
  line = f.readline()[:-1]
  if rinex_ver == 2:
    line = ' ' + line  # Shift 1 char to the right
  line = line.replace('D', 'E')  # Handle bizarro float format
  return float(line[4:23]), float(line[23:42]), float(line[42:61]), float(line[61:80])


def convert_ublox_gps_ephem(ublox_ephem, current_time: Optional[datetime] = None):
  # Week time of ephemeris gps msg has a roll-over period of 10 bits (19.6 years)
  # The latest roll-over was on 2019-04-07
  week = ublox_ephem.gpsWeek
  if current_time is None:
    # Each message is incremented to be greater or equal than week 1877 (2015-12-27).
    #  To skip this use the current_time argument
    week += 1024
    if week < 1877:
      week += 1024
  else:
    roll_overs = GPSTime.from_datetime(current_time).week // 1024
    week += (roll_overs - (week // 1024)) * 1024

  # GPS week refers to current week, the ephemeris can be valid for the next
  # if toe equals 0, this can be verified by the TOW count if it is within the
  # last 2 hours of the week (gps ephemeris valid for 4hours)
  if ublox_ephem.toe == 0 and ublox_ephem.towCount*6 >= (SECS_IN_WEEK - 2*SECS_IN_HR):
    week += 1

  ephem = {}
  ephem['sv_id'] = ublox_ephem.svId
  ephem['toe'] = GPSTime(week, ublox_ephem.toe)
  ephem['toc'] = GPSTime(week, ublox_ephem.toc)
  ephem['af0'] = ublox_ephem.af0
  ephem['af1'] = ublox_ephem.af1
  ephem['af2'] = ublox_ephem.af2
  ephem['tgd'] = ublox_ephem.tgd

  ephem['sqrta'] = np.sqrt(ublox_ephem.a)
  ephem['dn'] = ublox_ephem.deltaN
  ephem['m0'] = ublox_ephem.m0

  ephem['ecc'] = ublox_ephem.ecc
  ephem['w'] = ublox_ephem.omega
  ephem['cus'] = ublox_ephem.cus
  ephem['cuc'] = ublox_ephem.cuc
  ephem['crc'] = ublox_ephem.crc
  ephem['crs'] = ublox_ephem.crs
  ephem['cic'] = ublox_ephem.cic
  ephem['cis'] = ublox_ephem.cis

  ephem['inc'] = ublox_ephem.i0
  ephem['inc_dot'] = ublox_ephem.iDot
  ephem['omegadot'] = ublox_ephem.omegaDot
  ephem['omega0'] = ublox_ephem.omega0

  ephem['healthy'] = ublox_ephem.svHealth == 0.0

  epoch = ephem['toe']
  return GPSEphemeris(ephem, epoch)


def convert_ublox_glonass_ephem(ublox_ephem, current_time: Optional[datetime] = None):
  ephem = {}
  ephem['prn'] = 'R%02i' % ublox_ephem.svId

  etime = datetime.strptime(f"{ublox_ephem.year}-{ublox_ephem.dayInYear}", "%Y-%j")
  # glonass time: UTC + 3h
  time_in_day = timedelta(hours=ublox_ephem.hour, minutes=ublox_ephem.minute, seconds=ublox_ephem.second)
  ephem['toc'] = GPSTime.from_datetime(etime + time_in_day - timedelta(hours=3))
  ephem['toe'] = GPSTime.from_datetime(etime + timedelta(minutes=(ublox_ephem.tb*15 - 180)))

  ephem['x'] = ublox_ephem.x # km
  ephem['x_vel'] = ublox_ephem.xVel # km/s
  ephem['x_acc'] = ublox_ephem.xAccel # km/s*s

  ephem['y'] = ublox_ephem.y # km
  ephem['y_vel'] = ublox_ephem.yVel # km/s
  ephem['y_acc'] = ublox_ephem.yAccel # km/s*s

  ephem['z'] = ublox_ephem.z # km
  ephem['z_vel'] = ublox_ephem.zVel # km/s
  ephem['z_acc'] = ublox_ephem.zAccel # km/s*s

  ephem['healthy'] = ublox_ephem.svHealth == 0.0
  ephem['age'] = ublox_ephem.age # age of information [days]

  # tauN compared to ephemeris from gdc.cddis.eosdis.nasa.gov is times -1
  ephem['min_tauN'] = ublox_ephem.tauN * (-1) # time correction relative to GLONASS tc
  ephem['GammaN'] = ublox_ephem.gammaN

  # TODO: channel is in string 7, which is not parsed
  ephem['freq_num'] = "1"

  # NOTE: ublox_ephem.tk is in a different format than rinex tk
  return GLONASSEphemeris(ephem, ephem['toe'])


class EphemerisType(IntEnum):
  # Matches the enum in log.capnp
  NAV = 0
  FINAL_ORBIT = 1
  RAPID_ORBIT = 2
  ULTRA_RAPID_ORBIT = 3
  QCOM_POLY = 4

  @staticmethod
  def all_orbits():
    return EphemerisType.FINAL_ORBIT, EphemerisType.RAPID_ORBIT, EphemerisType.ULTRA_RAPID_ORBIT

  @classmethod
  def from_file_name(cls, file_name: str):
    if "/final" in file_name or "/igs" in file_name:
      return EphemerisType.FINAL_ORBIT
    if "/rapid" in file_name or "/igr" in file_name:
      return EphemerisType.RAPID_ORBIT
    if "/ultra" in file_name or "/igu" in file_name or "COD0OPSULT" in file_name:
      return EphemerisType.ULTRA_RAPID_ORBIT
    raise RuntimeError(f"Ephemeris type not found in filename: {file_name}")


class Ephemeris(ABC):

  def __init__(self, prn: str, data, epoch: GPSTime, eph_type: EphemerisType, healthy: bool, max_time_diff: float,
               file_epoch: Optional[GPSTime] = None, file_name=None):
    self.prn = prn
    self.data = data
    self.epoch = epoch
    self.eph_type = eph_type
    self.healthy = healthy
    self.max_time_diff = max_time_diff
    self.file_epoch = file_epoch
    self.file_source = '' if file_name is None else file_name.split('/')[-1][:3]  # File source for the ephemeris (e.g. igu, igr, Sta)
    self.__json = None

  def valid(self, time):
    return abs(time - self.epoch) <= self.max_time_diff

  def __repr__(self):
    time = self.epoch.as_datetime().strftime('%Y-%m-%dT%H:%M:%S.%f')
    return f"<{self.__class__.__name__} from {self.prn} at {time}>"

  def get_sat_info(self, time: GPSTime):
    """
    Returns: (pos, vel, clock_err, clock_rate_err, ephemeris)
    """
    if not self.healthy:
      return None
    return list(self._get_sat_info(time)) + [self]

  @abstractmethod
  def _get_sat_info(self, time):
    pass

  def to_json(self):
    if self.__json is None:
      dict = self.__dict__
      dict['ephemeris_class'] = self.__class__.__name__
      self.__json = {'ephemeris': json.dumps(dict, cls=EphemerisSerializer)}
    return self.__json

  @classmethod
  def from_json(cls, json_dct):
    dct = json.loads(json_dct['ephemeris'], object_hook=ephemeris_deserialize_hook)
    obj = cls.__new__(globals()[dct['ephemeris_class']])
    obj.__dict__.update(dct)
    obj.__json = json_dct
    return obj


def ephemeris_deserialize_hook(dct):
  if 'week' in dct:
    return GPSTime(dct['week'], dct['tow'])
  return dct


class EphemerisSerializer(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, GPSTime):
      return o.__dict__
    if isinstance(o, np.ndarray):
      return o.tolist()
    return json.JSONEncoder.default(self, o)


class GLONASSEphemeris(Ephemeris):
  def __init__(self, data, epoch, file_name=None):
    super().__init__(data['prn'], data, epoch, EphemerisType.NAV, data['healthy'], max_time_diff=25*SECS_IN_MIN, file_name=file_name)
    self.channel = data['freq_num']
    self.to_json()

  def _get_sat_info(self, time: GPSTime):
    # see the russian doc for this:
    # http://gauss.gge.unb.ca/GLONASS.ICD.pdf

    eph = self.data
    tdiff = time - utc_to_gpst(eph['toe'])

    # Clock correction (except for general relativity which is applied later)
    clock_err = eph['min_tauN'] + tdiff * (eph['GammaN'])
    clock_rate_err = eph['GammaN']

    def glonass_diff_eq(state, acc):
      J2 = 1.0826257e-3
      mu = 3.9860044e14
      omega = 7.292115e-5
      ae = 6378136.0
      r = np.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
      ders = np.zeros(6)
      if r**2 < 0:
        return ders
      a = 1.5 * J2 * mu * (ae**2)/ (r**5)
      b = 5 * (state[2]**2) / (r**2)
      c = -mu/(r**3) - a*(1-b)

      ders[0:3] = state[3:6]
      ders[3] = (c + omega**2)*state[0] + 2*omega*state[4] + acc[0]
      ders[4] = (c + omega**2)*state[1] - 2*omega*state[3] + acc[1]
      ders[5] = (c - 2*a)*state[2] + acc[2]
      return ders

    init_state = np.empty(6)
    init_state[0] = eph['x']
    init_state[1] = eph['y']
    init_state[2] = eph['z']
    init_state[3] = eph['x_vel']
    init_state[4] = eph['y_vel']
    init_state[5] = eph['z_vel']
    init_state = 1000*init_state
    acc = 1000*np.array([eph['x_acc'], eph['y_acc'], eph['z_acc']])
    state = init_state
    tstep = 90
    if tdiff < 0:
      tt = -tstep
    elif tdiff > 0:
      tt = tstep
    while abs(tdiff) > 1e-9:
      if abs(tdiff) < tstep:
        tt = tdiff
      k1 = glonass_diff_eq(state, acc)
      k2 = glonass_diff_eq(state + k1*tt/2, -acc)
      k3 = glonass_diff_eq(state + k2*tt/2, -acc)
      k4 = glonass_diff_eq(state + k3*tt, -acc)
      state += (k1 + 2*k2 + 2*k3 + k4)*tt/6.0
      tdiff -= tt

    pos = state[0:3]
    vel = state[3:6]
    return pos, vel, clock_err, clock_rate_err


class PolyEphemeris(Ephemeris):
  def __init__(self, prn: str, data, epoch: GPSTime, ephem_type: EphemerisType,
               file_epoch: GPSTime=None, file_name: str=None, healthy=True, tgd=0,
               max_time_diff: int=SECS_IN_HR):
    super().__init__(prn, data, epoch, ephem_type, healthy, max_time_diff=max_time_diff, file_epoch=file_epoch, file_name=file_name)
    self.tgd = tgd
    self.to_json()

  def _get_sat_info(self, time: GPSTime):
    dt = time - self.data['t0']
    deg = self.data['deg']
    deg_t = self.data['deg_t']
    indices = np.arange(deg+1)[:,np.newaxis]
    sat_pos = np.sum((dt**indices)*self.data['xyz'], axis=0)
    indices = indices[1:]
    sat_vel = np.sum(indices*(dt**(indices-1)*self.data['xyz'][1:]), axis=0)
    time_err = sum((dt**p)*self.data['clock'][deg_t-p] for p in range(deg_t+1))
    time_err_rate = sum(p*(dt**(p-1))*self.data['clock'][deg_t-p] for p in range(1,deg_t+1))
    time_err_with_rel = time_err - 2*np.inner(sat_pos, sat_vel)/SPEED_OF_LIGHT**2
    return sat_pos, sat_vel, time_err_with_rel, time_err_rate


class GPSEphemeris(Ephemeris):
  def __init__(self, data, epoch, file_name=None):
    super().__init__('G%02i' % data['sv_id'], data, epoch, EphemerisType.NAV, data['healthy'], max_time_diff=2*SECS_IN_HR, file_name=file_name)
    self.max_time_diff_tgd = SECS_IN_DAY
    self.to_json()

  def get_tgd(self):
    return self.data['tgd']

  def _get_sat_info(self, time: GPSTime):
    eph = self.data
    tdiff = time - eph['toc']  # Time of clock
    clock_err = eph['af0'] + tdiff * (eph['af1'] + tdiff * eph['af2'])
    clock_rate_err = eph['af1'] + 2 * tdiff * eph['af2']

    # Orbit propagation
    tdiff = time - eph['toe']  # Time of ephemeris (might be different from time of clock)

    # Calculate position per IS-GPS-200D p 97 Table 20-IV
    a = eph['sqrta'] * eph['sqrta']  # [m] Semi-major axis
    ma_dot = sqrt(EARTH_GM / (a * a * a)) + eph['dn']  # [rad/sec] Corrected mean motion
    ma = eph['m0'] + ma_dot * tdiff  # [rad] Corrected mean anomaly

    # Iteratively solve for the Eccentric Anomaly (from Keith Alter and David Johnston)
    ea = ma  # Starting value for E

    ea_old = 2222
    while fabs(ea - ea_old) > 1.0E-14:
      ea_old = ea
      tempd1 = 1.0 - eph['ecc'] * cos(ea_old)
      ea = ea + (ma - ea_old + eph['ecc'] * sin(ea_old)) / tempd1
    ea_dot = ma_dot / tempd1

    # Relativistic correction term
    einstein = -4.442807633E-10 * eph['ecc'] * eph['sqrta'] * sin(ea)

    # Begin calc for True Anomaly and Argument of Latitude
    tempd2 = sqrt(1.0 - eph['ecc'] * eph['ecc'])
    # [rad] Argument of Latitude = True Anomaly + Argument of Perigee
    al = atan2(tempd2 * sin(ea), cos(ea) - eph['ecc']) + eph['w']
    al_dot = tempd2 * ea_dot / tempd1

    # Calculate corrected argument of latitude based on position
    cal = al + eph['cus'] * sin(2.0 * al) + eph['cuc'] * cos(2.0 * al)
    cal_dot = al_dot * (1.0 + 2.0 * (eph['cus'] * cos(2.0 * al) -
                                     eph['cuc'] * sin(2.0 * al)))

    # Calculate corrected radius based on argument of latitude
    r = a * tempd1 + eph['crc'] * cos(2.0 * al) + eph['crs'] * sin(2.0 * al)
    r_dot = (a * eph['ecc'] * sin(ea) * ea_dot +
             2.0 * al_dot * (eph['crs'] * cos(2.0 * al) -
                             eph['crc'] * sin(2.0 * al)))

    # Calculate inclination based on argument of latitude
    inc = (eph['inc'] + eph['inc_dot'] * tdiff +
           eph['cic'] * cos(2.0 * al) +
           eph['cis'] * sin(2.0 * al))
    inc_dot = (eph['inc_dot'] +
               2.0 * al_dot * (eph['cis'] * cos(2.0 * al) -
                               eph['cic'] * sin(2.0 * al)))

    # Calculate position and velocity in orbital plane
    x = r * cos(cal)
    y = r * sin(cal)
    x_dot = r_dot * cos(cal) - y * cal_dot
    y_dot = r_dot * sin(cal) + x * cal_dot

    # Corrected longitude of ascending node
    om_dot = eph['omegadot'] - EARTH_ROTATION_RATE
    om = eph['omega0'] + tdiff * om_dot - EARTH_ROTATION_RATE * eph['toe'].tow

    # Compute the satellite's position in Earth-Centered Earth-Fixed coordinates
    pos = np.empty(3)
    pos[0] = x * cos(om) - y * cos(inc) * sin(om)
    pos[1] = x * sin(om) + y * cos(inc) * cos(om)
    pos[2] = y * sin(inc)

    tempd3 = y_dot * cos(inc) - y * sin(inc) * inc_dot

    # Compute the satellite's velocity in Earth-Centered Earth-Fixed coordinates
    vel = np.empty(3)
    vel[0] = -om_dot * pos[1] + x_dot * cos(om) - tempd3 * sin(om)
    vel[1] = om_dot * pos[0] + x_dot * sin(om) + tempd3 * cos(om)
    vel[2] = y * cos(inc) * inc_dot + y_dot * sin(inc)

    clock_err += einstein

    return pos, vel, clock_err, clock_rate_err


def parse_sp3_orbits(file_names, supported_constellations, skip_until_epoch: Optional[GPSTime] = None) -> Dict[str, List[PolyEphemeris]]:
  if skip_until_epoch is None:
    skip_until_epoch = GPSTime(0, 0)
  data: Dict[str, List] = {}
  for file_name in file_names:
    if file_name is None:
      continue
    with open(file_name) as f:
      ephem_type = EphemerisType.from_file_name(file_name)
      file_epoch = None
      while True:
        line = f.readline()[:-1]
        if not line:
          break
        # epoch header
        if line[0:2] == '* ':
          year = int(line[3:7])
          month = int(line[8:10])
          day = int(line[11:13])
          hour = int(line[14:16])
          minute = int(line[17:19])
          second = int(float(line[20:31]))
          epoch = GPSTime.from_datetime(datetime(year, month, day, hour, minute, second))
          if file_epoch is None:
            file_epoch = epoch
        # pos line
        elif line[0] == 'P':
          # Skipping data can reduce the time significantly when parsing the ephemeris
          if epoch < skip_until_epoch:
            continue
          prn = line[1:4].replace(' ', '0')
          # In old SP3 files vehicle ID doesn't contain constellation
          # identifier. We assume that constellation is GPS when missing.
          if prn[0] == '0':
            prn = 'G' + prn[1:]
          if get_constellation(prn) not in supported_constellations:
            continue
          if prn not in data:
            data[prn] = []
          #TODO this is a crappy way to deal with overlapping ultra rapid
          if len(data[prn]) < 1 or epoch - data[prn][-1][1] > 0:
            parsed = [(ephem_type, file_epoch, file_name),
                      epoch,
                      1e3 * float(line[4:18]),
                      1e3 * float(line[18:32]),
                      1e3 * float(line[32:46]),
                      1e-6 * float(line[46:60])]
            if (np.array(parsed[2:]) != 0).all():
              data[prn].append(parsed)
  ephems = {}
  for prn in data:
    ephems[prn] = read_prn_data(data, prn)
  return ephems


def read_prn_data(data, prn, deg=16, deg_t=1):
  np_data_prn = np.array(data[prn], dtype=object)
  # Currently, don't even bother with satellites that have unhealthy times
  if len(np_data_prn) == 0 or (np_data_prn[:, 5] > .99).any():
    return []
  ephems = []
  for i in range(len(np_data_prn) - deg):
    epoch_index = i + deg // 2
    epoch = np_data_prn[epoch_index][1]
    measurements = np_data_prn[i:i + deg + 1, 1:5]

    times = (measurements[:, 0] - epoch).astype(float)
    if not (np.diff(times) != 900).any() and not (np.diff(times) != 300).any():
      continue

    poly_data = {}
    poly_data['t0'] = epoch
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # Ignores: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
      poly_data['xyz'] = poly.polyfit(times, measurements[:, 1:].astype(float), deg)
    poly_data['clock'] = [(np_data_prn[epoch_index + 1][5] - np_data_prn[epoch_index - 1][5]) / 1800, np_data_prn[epoch_index][5]]
    poly_data['deg'] = deg
    poly_data['deg_t'] = deg_t
    # It can happen that a mix of orbit ephemeris types are used in the polyfit.
    ephem_type, file_epoch, file_name = np_data_prn[epoch_index][0]
    ephems.append(PolyEphemeris(prn, poly_data, epoch, ephem_type, file_epoch, file_name, healthy=True))
  return ephems


def parse_rinex_nav_msg_gps(file_name):
  ephems = defaultdict(list)
  got_header = False
  rinex_ver = None
  #ion_alpha = None
  #ion_beta = None
  f = open(file_name)
  while True:
    line = f.readline()[:-1]
    if not line:
      break
    if not got_header:
      if rinex_ver is None:
        if line[60:80] != "RINEX VERSION / TYPE":
          raise RuntimeError("Doesn't appear to be a RINEX file")
        rinex_ver = int(float(line[0:9]))
        if line[20] != "N":
          raise RuntimeError("Doesn't appear to be a Navigation Message file")
      #if line[60:69] == "ION ALPHA":
      #  line = line.replace('D', 'E')  # Handle bizarro float format
      #  ion_alpha= [float(line[3:14]), float(line[15:26]), float(line[27:38]), float(line[39:50])]
      #if line[60:68] == "ION BETA":
      #  line = line.replace('D', 'E')  # Handle bizarro float format
      #  ion_beta= [float(line[3:14]), float(line[15:26]), float(line[27:38]), float(line[39:50])]
      if line[60:73] == "END OF HEADER":
        #ion = ion_alpha + ion_beta
        got_header = True
      continue
    if rinex_ver == 3:
      if line[0] != 'G':
        continue
    if rinex_ver == 3:
      sv_id = int(line[1:3])
      epoch = GPSTime.from_datetime(datetime.strptime(line[4:23], "%y %m %d %H %M %S"))
    elif rinex_ver == 2:
      sv_id = int(line[0:2])
      # 2000 year is in RINEX file as 0, but Python requires two digit year: 00
      epoch_str = line[3:20]
      if epoch_str[0] == ' ':
        epoch_str = '0' + epoch_str[1:]
      epoch = GPSTime.from_datetime(datetime.strptime(epoch_str, "%y %m %d %H %M %S"))
      line = ' ' + line  # Shift 1 char to the right

    line = line.replace('D', 'E')  # Handle bizarro float format
    e = {'epoch': epoch, 'sv_id': sv_id}
    e['toc'] = epoch
    e['af0'] = float(line[23:42])
    e['af1'] = float(line[42:61])
    e['af2'] = float(line[61:80])

    e['iode'], e['crs'], e['dn'], e['m0'] = read4(f, rinex_ver)
    e['cuc'], e['ecc'], e['cus'], e['sqrta'] = read4(f, rinex_ver)
    toe_tow, e['cic'], e['omega0'], e['cis'] = read4(f, rinex_ver)
    e['inc'], e['crc'], e['w'], e['omegadot'] = read4(f, rinex_ver)
    e['inc_dot'], e['l2_codes'], toe_week, e['l2_pflag'] = read4(f, rinex_ver)
    e['sv_accuracy'], e['health'], e['tgd'], e['iodc'] = read4(f, rinex_ver)
    f.readline()  # Discard last row

    e['toe'] = GPSTime(toe_week, toe_tow)
    e['healthy'] = (e['health'] == 0.0)

    ephem = GPSEphemeris(e, epoch, file_name=file_name)
    ephems[ephem.prn].append(ephem)
  f.close()
  return ephems


def parse_rinex_nav_msg_glonass(file_name):
  ephems = defaultdict(list)
  f = open(file_name)
  got_header = False
  rinex_ver = None
  while True:
    line = f.readline()[:-1]
    if not line:
      break
    if not got_header:
      if rinex_ver is None:
        if line[60:80] != "RINEX VERSION / TYPE":
          raise RuntimeError("Doesn't appear to be a RINEX file")
        rinex_ver = int(float(line[0:9]))
        if line[20] != "G":
          raise RuntimeError("Doesn't appear to be a Navigation Message file")
      if line[60:73] == "END OF HEADER":
        got_header = True
      continue
    if rinex_ver == 3:
      prn = line[:3]
      epoch = GPSTime.from_datetime(datetime.strptime(line[4:23], "%y %m %d %H %M %S"))
    elif rinex_ver == 2:
      prn = 'R%02i' % int(line[0:2])
      epoch = GPSTime.from_datetime(datetime.strptime(line[3:20], "%y %m %d %H %M %S"))
      line = ' ' + line  # Shift 1 char to the right

    line = line.replace('D', 'E')  # Handle bizarro float format
    e = {'epoch': epoch, 'prn': prn}
    e['toe'] = epoch
    e['min_tauN'] = float(line[23:42])
    e['GammaN'] = float(line[42:61])
    e['tk'] = float(line[61:80])

    e['x'], e['x_vel'], e['x_acc'], e['health'] = read4(f, rinex_ver)
    e['y'], e['y_vel'], e['y_acc'], e['freq_num'] = read4(f, rinex_ver)
    e['z'], e['z_vel'], e['z_acc'], e['age'] = read4(f, rinex_ver)

    e['healthy'] = (e['health'] == 0.0)
    ephems[prn].append(GLONASSEphemeris(e, epoch, file_name=file_name))
  f.close()
  return ephems


def parse_qcom_ephem(qcom_poly, current_week):
  svId = qcom_poly.svId
  data = qcom_poly
  t0 = data.t0
  # fix glonass time
  prn = get_prn_from_nmea_id(svId)
  if prn == 'GLONASS':
    # TODO should handle leap seconds better
    epoch = GPSTime(current_week, (t0 + 3*SECS_IN_WEEK) % (SECS_IN_WEEK) + 18)
  else:
    epoch = GPSTime(current_week, t0)
  poly_data = {}
  poly_data['t0'] = epoch
  poly_data['xyz'] = np.array([
                      [data.xyz0[0], data.xyzN[0], data.xyzN[1], data.xyzN[2]],
                      [data.xyz0[1], data.xyzN[3], data.xyzN[4], data.xyzN[5]],
                      [data.xyz0[2], data.xyzN[6], data.xyzN[7], data.xyzN[8]] ]).T

  poly_data['clock'] = [1e-3*data.other[3], 1e-3*data.other[2], 1e-3*data.other[1], 1e-3*data.other[0]]
  poly_data['deg'] = 3
  poly_data['deg_t'] = 3
  return PolyEphemeris(prn, poly_data, epoch, ephem_type=EphemerisType.QCOM_POLY, max_time_diff=180)
