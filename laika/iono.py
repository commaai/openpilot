import datetime as dt
import numpy as np
import re
from math import cos, sin, pi, floor
from .constants import SECS_IN_MIN, SECS_IN_HR, EARTH_RADIUS
from .lib.coordinates import LocalCoord
from .gps_time import GPSTime

# Altitude of Ionospheric-pierce-point
IPP_ALT = 6821000


def closest_in_list(lst, val, num=2):
  """
    Returns two (`num` in general) closest values of `val` in list `lst`
    """
  idxs = sorted(lst, key=lambda x: abs(x - val))[:num]
  return sorted(list(lst).index(x) for x in idxs)


def get_header_line(headr, proprty):
    """
    :param headr: the header of the RINEX-file
    :param proprty: string-like property to search for (e.g. 'delta-utc')
    :return: the string of the ``headr`` containing ``property``
    """
    pattern = re.compile(proprty, re.IGNORECASE)
    for d in headr:
        if pattern.search(d):
            return d


def get_header_body(file_path):
    """
    Opens `file_path`, reads file and returns header and body
    separated with "END OF HEADER"
    :param file_path: path to RINEX-like file
    :return: header, body (arrays of lines)
    """
    with open(file_path) as fd:
        data = fd.readlines()
        for j, d in enumerate(data):
            if "END OF HEADER" in d:
                header_end = j
                break
    return data[:header_end], data[header_end + 1:]


def get_int_from_header(hdr, seq):
    """
    Returns the first int from the line that contains `seq` of lines `hdr`.
    In fact, _header_ here may not be header of RINEX/IONEX, just some set of lines.
    """
    return int(get_header_line(hdr, seq).split()[0])

def compute_grid_lats_lons(data):
  grid = np.array([], dtype='uint16')
  lats = np.array([])
  for j, line in enumerate(data[1:]):
    if "LAT" in line:
      lat, lon1, lon2, dlon, h = (float(line[x:x + 6]) for x in range(2, 32, 6))
      lats = np.append(lats, lat)
      row_length = (lon2 - lon1) / dlon + 1  # total number of values of longitudes
      next_lines_with_numbers = int(np.ceil(row_length / 16))
      elems_in_row = [
        min(16, int(row_length - i * 16)) for i in range(next_lines_with_numbers)
      ]
      row = np.array([], dtype='int16')
      for i, elem in enumerate(elems_in_row):
        row = np.append(
          row,
          np.array(
            [int(data[j + 2 + i][5 * x:5 * x + 5]) for x in range(elem)],
            dtype='int16',
          ),
        )
      if len(grid) > 0:
        grid = np.vstack((grid, row))
      else:
        grid = np.append(grid, row)
  lons = np.linspace(lon1, lon2, int(row_length))
  return (grid, lats, lons)


class IonexMap:
  def __init__(self, exp, data1, data2):
    self.exp = exp
    self.t1 = GPSTime.from_datetime(dt.datetime(*[int(d) for d in data1[0].split()[:6]]))
    self.t2 = GPSTime.from_datetime(dt.datetime(*[int(d) for d in data2[0].split()[:6]]))
    assert self.t2 - self.t1 == SECS_IN_HR
    assert len(data1) == len(data2)

    self.max_time_diff = SECS_IN_MIN*30
    self.epoch = self.t1 + self.max_time_diff

    self.grid_TEC1, self.lats, self.lons = compute_grid_lats_lons(data1)
    self.grid_TEC2, self.lats, self.lons = compute_grid_lats_lons(data2)

  def valid(self, time):
    return abs(time - self.epoch) <= self.max_time_diff

  @staticmethod
  def find_nearest(lst, val):
    return (np.abs(lst - val)).argmin()

  def get_TEC(self, pos, time):
    """
        Returns TEC in a position `pos`  of ionosphere
        :param pos: (lat, lon) [deg, deg]
        :return:
        """
    if pos[0] in self.lats and pos[1] in self.lons:
      lat = self.find_nearest(self.lats, pos[0])
      lon = self.find_nearest(self.lons, pos[1])
      E = self.grid_TEC1[lat][lon] + self.grid_TEC2[lat][lon]
      return E
    lat_idxs = closest_in_list(self.lats, pos[0])
    lon_idxs = closest_in_list(self.lons, pos[1])
    lat0, lat1 = self.lats[lat_idxs[0]], self.lats[lat_idxs[1]]
    lon0, lon1 = self.lons[lon_idxs[0]], self.lons[lon_idxs[1]]
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    p = float(pos[0] - lat0) / dlat
    q = float(pos[1] - lon0) / dlon

    (E00, E10), (E01, E11) = self.grid_TEC1[lat_idxs[0]:lat_idxs[1] + 1, lon_idxs[0]:lon_idxs[1] + 1]
    TEC_1 = ((1 - p) * (1 - q) * E00 + p * (1 - q) * E01 + (1 - p) * q * E10 + p * q * E11)
    (E00, E10), (E01, E11) = self.grid_TEC2[lat_idxs[0]:lat_idxs[1] + 1, lon_idxs[0]:lon_idxs[1] + 1]
    TEC_2 = ((1 - p) * (1 - q) * E00 + p * (1 - q) * E01 + (1 - p) * q * E10 + p * q * E11)

    return (1 - (time - self.t1)/SECS_IN_HR)*TEC_1 + ((time - self.t1)/SECS_IN_HR)*TEC_2

  def get_delay(self, rcv_pos, az, el, sat_pos, time, freq):
    # To get a delay from a TEC map, we need to calculate
    # the ionospheric pierce point, geometry described here
    # https://en.wikipedia.org/wiki/Ionospheric_pierce_point
    conv = LocalCoord.from_ecef(rcv_pos)
    geocentric_alt = np.linalg.norm(rcv_pos)
    alpha = np.pi/2 + el
    beta = np.arcsin(geocentric_alt*np.sin(alpha)/IPP_ALT)
    gamma = np.pi - alpha - beta
    ipp_dist = geocentric_alt*np.sin(gamma)/np.sin(beta)
    ipp_ned = conv.ecef2ned(sat_pos)*(ipp_dist)/np.linalg.norm(sat_pos)
    ipp_geo = conv.ned2geodetic(ipp_ned)
    factor = 40.30E16 / (freq**2) * 10**(self.exp)
    vertical_delay = self.get_TEC(ipp_geo, time) * factor
    slant_delay = vertical_delay * ((1 - ((EARTH_RADIUS * np.sin(beta)) /
                                          (EARTH_RADIUS + 3.5e5))**2)**(-0.5))
    return slant_delay

  @staticmethod
  def round_to_grid(number, base):
    return int(base * round(float(number) / base))


def parse_ionex(ionex_file):
  """
    :param ionex_file: path to the IONEX file
    :return: TEC interpolation function `f( (lat,lon), datetime )`
    """
  header, body = get_header_body(ionex_file)

  exponent = get_int_from_header(header, "EXPONENT")
  maps_count = get_int_from_header(header, "MAPS IN FILE")
  # =============
  # Separate maps
  # =============
  map_start_idx = []
  map_end_idx = []

  for j, line in enumerate(body):
    if "START OF TEC MAP" in line:
      map_start_idx += [j]
    elif "END OF TEC MAP" in line:
      map_end_idx += [j]
  if maps_count != len(map_start_idx):
    raise LookupError("Parsing error: the number of maps in the header "
                      "is not equal to the number of maps in the body.")
  if len(map_start_idx) != len(map_end_idx):
    raise IndexError("Starts end ends numbers are not equal.")
  map_dates = []
  for i in range(maps_count):
    date_components = body[map_start_idx[i] + 1].split()[:6]
    map_dates.append(dt.datetime(*[int(d) for d in date_components]))

  maps = []
  iono_map = iono_map_prev = None
  for m in range(maps_count):
    iono_map_prev = iono_map
    iono_map = body[map_start_idx[m] + 1:map_end_idx[m]]
    if iono_map and iono_map_prev:
      maps += [IonexMap(exponent, iono_map_prev, iono_map)]
  return maps


def klobuchar(pos, az, el, time, iono_coeffs):
  """
    Details are taken from [5]: IS-GPS-200H, Fig. 20-4
    Note: result is referred to the GPS L₁ frequency;
    if the user is operating on the GPS L₂ frequency, the correction term must
    be multiplied by γ = f₂²/f₁¹ = 0.6071850227694382
    :param pos: [lat, lon, alt] in radians and meters
    """

  tow = time.tow
  if pos[2] < -1E3 or el < 0:
    return 0.0
  if len(iono_coeffs) < 8:
    return None

  # earth centered angle (semi-circle)
  psi = 0.0137 / (el / pi + 0.11) - 0.022

  # subionospheric latitude/longitude (semi-circle)
  phi = pos[0] / pi + psi * cos(az)
  if phi > 0.416:
    phi = 0.416
  elif phi < -0.416:
    phi = -0.416
  lam = pos[1] / pi + psi * sin(az) / cos(phi * pi)

  # geomagnetic latitude (semi-circle) */
  phi += 0.064 * cos((lam - 1.617) * pi)

  # local time (s)
  tt = 43200.0 * lam + tow
  tt -= floor(tt / 86400.0) * 86400.0  # 0<=tt<86400

  # slant factor
  f = 1.0 + 16.0 * pow(0.53 - el / pi, 3.0)

  # ionospheric delay
  amp = iono_coeffs[0] + phi * (iono_coeffs[1] + phi *
                                (iono_coeffs[2] + phi * iono_coeffs[3]))
  per = iono_coeffs[4] + phi * (iono_coeffs[5] + phi *
                                (iono_coeffs[6] + phi * iono_coeffs[7]))
  if amp < 0.0:
    amp = 0.
  if per < 72000.0:
    per = 72000.0
  x = 2.0 * pi * (tt - 50400.0) / per

  mul = 5E-9
  if abs(x) < 1.57:
    mul = (5E-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0)))
  return 2.99792458E8 * f * mul
