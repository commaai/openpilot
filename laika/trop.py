#!/usr/bin/python

from numpy import cos, exp, pi
from .lib.coordinates import ecef2geodetic


def saast(pos, el, humi=0.75, temp0=15.0):
  """
    Function from RTKlib: https://github.com/tomojitakasu/RTKLIB/blob/master/src/rtkcmn.c#L3362-3362
        with no changes
    :param time:    time
    :param pos:     receiver position {ecef} m)
    :param el:    azimuth/elevation angle {az,el} (rad) -- we do not use az
    :param humi:    relative humidity
    :param temp0:   temperature (Celsius)
    :return:        tropospheric delay (m)
    """
  pos_rad = ecef2geodetic(pos, radians=True)
  if pos_rad[2] < -1E3 or 1E4 < pos_rad[2] or el <= 0:
    return 0.0

  # /* standard atmosphere */
  hgt = 0.0 if pos_rad[2] < 0.0 else pos_rad[2]

  pres = 1013.25 * pow(1.0 - 2.2557E-5 * hgt, 5.2568)
  temp = temp0 - 6.5E-3 * hgt + 273.16
  e = 6.108 * humi * exp((17.15 * temp - 4684.0) / (temp - 38.45))

  # /* saastamoninen model */
  z = pi / 2.0 - el
  trph = 0.0022768 * pres / (
    1.0 - 0.00266 * cos(2.0 * pos_rad[0]) - 0.00028 * hgt / 1E3) / cos(z)
  trpw = 0.002277 * (1255.0 / temp + 0.05) * e / cos(z)
  return trph + trpw
