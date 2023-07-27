import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
import numpy.polynomial.polynomial as poly
from datetime import datetime
from math import sin, cos, sqrt, fabs, atan2

from .gps_time import GPSTime, utc_to_gpst
from .constants import SPEED_OF_LIGHT, SECS_IN_MIN, SECS_IN_HR, SECS_IN_DAY, \
                       EARTH_ROTATION_RATE, EARTH_GM
from .helpers import get_constellation, get_prn_from_nmea_id

import capnp
import os
capnp.remove_import_hook()
capnp_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ephemeris.capnp"))
ephemeris_structs = capnp.load(capnp_path)


def read4(f, rinex_ver):
  line = f.readline()[:-1]
  if rinex_ver == 2:
    line = ' ' + line  # Shift 1 char to the right
  line = line.replace('D', 'E')  # Handle bizarro float format
  return float(line[4:23]), float(line[23:42]), float(line[42:61]), float(line[61:80])


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

  def __init__(self, prn: str, epoch: GPSTime, eph_type: EphemerisType, healthy: bool, max_time_diff: float,
               file_epoch: Optional[GPSTime] = None, file_name=None):
    self.prn = prn
    self.epoch = epoch
    self.eph_type = eph_type
    self.healthy = healthy
    self.max_time_diff = max_time_diff
    self.file_epoch = file_epoch
    self.file_name = file_name
    self.file_source = '' if file_name is None else file_name.split('/')[-1][:3]  # File source for the ephemeris (e.g. igu, igr, Sta)

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


class GLONASSEphemeris(Ephemeris):
  def __init__(self, data, file_name=None):
    self.epoch = GPSTime.from_glonass(data.n4, data.nt, data.tb*15*SECS_IN_MIN)
    super().__init__('R%02i' % data.svId, self.epoch, EphemerisType.NAV, data.svHealth==0, max_time_diff=25*SECS_IN_MIN, file_name=file_name)
    self.data = data
    self.epoch =  GPSTime.from_glonass(data.n4, data.nt, data.tb*15 * SECS_IN_MIN)
    self.channel = data.freqNum

  def _get_sat_info(self, time: GPSTime):
    # see the russian doc for this:
    # http://gauss.gge.unb.ca/GLONASS.ICD.pdf

    eph = self.data
    tdiff = time - self.epoch
    # Clock correction (except for general relativity which is applied later)
    clock_err = -eph.tauN + tdiff * eph.gammaN
    clock_rate_err = eph.gammaN

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
    init_state[0] = eph.x
    init_state[1] = eph.y
    init_state[2] = eph.z
    init_state[3] = eph.xVel
    init_state[4] = eph.yVel
    init_state[5] = eph.zVel
    init_state = 1000*init_state
    acc = 1000*np.array([eph.xAccel, eph.yAccel, eph.zAccel])
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
               file_epoch: Optional[GPSTime] = None, file_name: Optional[str] = None, healthy=True, tgd=0,
               max_time_diff: int=SECS_IN_HR):
    super().__init__(prn, epoch, ephem_type, healthy, max_time_diff=max_time_diff, file_epoch=file_epoch, file_name=file_name)
    self.data = data
    self.tgd = tgd

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
  def __init__(self, data, file_name=None):
    self.toe = GPSTime(data.toeWeek, data.toe)
    self.toc = GPSTime(data.tocWeek, data.toc)
    self.epoch = self.toc

    super().__init__('G%02i' % data.svId, self.epoch, EphemerisType.NAV, data.svHealth==0, max_time_diff=2*SECS_IN_HR, file_name=file_name)
    self.max_time_diff_tgd = SECS_IN_DAY
    self.data = data
    self.sqrta = np.sqrt(data.a)

  def get_tgd(self):
    return self.datatgd

  def _get_sat_info(self, time: GPSTime):
    eph = self.data
    tdiff = time - self.toc  # Time of clock
    clock_err = eph.af0 + tdiff * (eph.af1 + tdiff * eph.af2)
    clock_rate_err = eph.af1 + 2 * tdiff * eph.af2\

    # Orbit propagation
    tdiff = time - self.toe  # Time of ephemeris (might be different from time of clock)

    # Calculate position per IS-GPS-200D p 97 Table 20-IV
    a = self.sqrta * self.sqrta # [m] Semi-major axis
    ma_dot = sqrt(EARTH_GM / (a * a * a)) + eph.deltaN  # [rad/sec] Corrected mean motion
    ma = eph.m0 + ma_dot * tdiff  # [rad] Corrected mean anomaly

    # Iteratively solve for the Eccentric Anomaly (from Keith Alter and David Johnston)
    ea = ma  # Starting value for E

    ea_old = 2222
    while fabs(ea - ea_old) > 1.0E-14:
      ea_old = ea
      tempd1 = 1.0 - eph.ecc * cos(ea_old)
      ea = ea + (ma - ea_old + eph.ecc * sin(ea_old)) / tempd1
    ea_dot = ma_dot / tempd1

    # Relativistic correction term
    einstein = -4.442807633E-10 * eph.ecc * self.sqrta * sin(ea)

    # Begin calc for True Anomaly and Argument of Latitude
    tempd2 = sqrt(1.0 - eph.ecc * eph.ecc)
    # [rad] Argument of Latitude = True Anomaly + Argument of Perigee
    al = atan2(tempd2 * sin(ea), cos(ea) - eph.ecc) + eph.omega
    al_dot = tempd2 * ea_dot / tempd1

    # Calculate corrected argument of latitude based on position
    cal = al + eph.cus * sin(2.0 * al) + eph.cuc * cos(2.0 * al)
    cal_dot = al_dot * (1.0 + 2.0 * (eph.cus * cos(2.0 * al) -
                                     eph.cuc * sin(2.0 * al)))

    # Calculate corrected radius based on argument of latitude
    r = a * tempd1 + eph.crc * cos(2.0 * al) + eph.crs * sin(2.0 * al)
    r_dot = (a * eph.ecc * sin(ea) * ea_dot +
             2.0 * al_dot * (eph.crs * cos(2.0 * al) -
                             eph.crc * sin(2.0 * al)))

    # Calculate inclination based on argument of latitude
    inc = (eph.i0 + eph.iDot * tdiff +
           eph.cic * cos(2.0 * al) +
           eph.cis * sin(2.0 * al))
    inc_dot = (eph.iDot +
               2.0 * al_dot * (eph.cis * cos(2.0 * al) -
                               eph.cic * sin(2.0 * al)))

    # Calculate position and velocity in orbital plane
    x = r * cos(cal)
    y = r * sin(cal)
    x_dot = r_dot * cos(cal) - y * cal_dot
    y_dot = r_dot * sin(cal) + x * cal_dot

    # Corrected longitude of ascending node
    om_dot = eph.omegaDot - EARTH_ROTATION_RATE
    om = eph.omega0 + tdiff * om_dot - EARTH_ROTATION_RATE * self.toe.tow

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
    e = {'svId': sv_id}
    # TODO are TOC and TOE the same?
    e['toc'] = epoch.tow
    e['tocWeek'] = epoch.week
    e['af0'] = float(line[23:42])
    e['af1'] = float(line[42:61])
    e['af2'] = float(line[61:80])

    e['iode'], e['crs'], e['deltaN'], e['m0'] = read4(f, rinex_ver)
    e['cuc'], e['ecc'], e['cus'], sqrta = read4(f, rinex_ver)
    e['a'] = sqrta ** 2
    e['toe'], e['cic'], e['omega0'], e['cis'] = read4(f, rinex_ver)
    e['i0'], e['crc'], e['omega'], e['omegaDot'] = read4(f, rinex_ver)
    e['iDot'], e['codesL2'], e['toeWeek'], l2_pflag = read4(f, rinex_ver)
    e['svAcc'], e['svHealth'], e['tgd'], e['iodc'] = read4(f, rinex_ver)
    f.readline()  # Discard last row

    data_struct = ephemeris_structs.Ephemeris.new_message(**e)

    ephem = GPSEphemeris(data_struct, file_name=file_name)
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
      sv_id = int(line[1:3])

      epoch = utc_to_gpst(GPSTime.from_datetime(datetime.strptime(line[4:23], "%y %m %d %H %M %S")))
    elif rinex_ver == 2:
      sv_id = int(line[0:2])
      epoch = utc_to_gpst(GPSTime.from_datetime(datetime.strptime(line[3:20], "%y %m %d %H %M %S")))
      line = ' ' + line  # Shift 1 char to the right

    line = line.replace('D', 'E')  # Handle bizarro float format
    e = {'svId': sv_id}
    e['n4'], e['nt'], toe_seconds = epoch.as_glonass()
    tb = toe_seconds / (15 * SECS_IN_MIN)

  
    e['tb'] = tb

    e['tauN'] = -float(line[23:42])
    e['gammaN'] = float(line[42:61])
    e['tkSeconds'] = float(line[61:80])

    e['x'], e['xVel'], e['xAccel'], e['svHealth'] = read4(f, rinex_ver)
    e['y'], e['yVel'], e['yAccel'], e['freqNum'] = read4(f, rinex_ver)
    e['z'], e['zVel'], e['zAccel'], e['age'] = read4(f, rinex_ver)
    
    # TODO unclear why glonass sometimes has nav messages 3s after correct one
    if abs(tb - int(tb)) > 1e-3:
      continue

    
    data_struct = ephemeris_structs.GlonassEphemeris.new_message(**e)
    ephem = GLONASSEphemeris(data_struct, file_name=file_name)

    ephems[ephem.prn].append(ephem)
  f.close()
  return ephems


def parse_qcom_ephem(qcom_poly):
  svId = qcom_poly.svId
  prn = get_prn_from_nmea_id(svId)
  epoch = GPSTime(qcom_poly.gpsWeek, qcom_poly.gpsTow)

  data = qcom_poly
  poly_data = {}
  poly_data['t0'] = epoch
  poly_data['xyz'] = np.array([
                      [data.xyz0[0], data.xyzN[0], data.xyzN[1], data.xyzN[2]],
                      [data.xyz0[1], data.xyzN[3], data.xyzN[4], data.xyzN[5]],
                      [data.xyz0[2], data.xyzN[6], data.xyzN[7], data.xyzN[8]] ]).T

  poly_data['clock'] = [1e-3*data.other[3], 1e-3*data.other[2], 1e-3*data.other[1], 1e-3*data.other[0]]
  poly_data['deg'] = 3
  poly_data['deg_t'] = 3
  return PolyEphemeris(prn, poly_data, epoch, ephem_type=EphemerisType.QCOM_POLY, max_time_diff=300, file_name='qcom')
