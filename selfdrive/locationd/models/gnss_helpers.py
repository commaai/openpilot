import numpy as np


# source: GNSSMeasurement (https://github.com/commaai/laika/blob/master/laika/raw_gnss.py)
class RawGNSSMeasurementIndices:
  PRN = 0
  RECV_TIME_WEEK = 1
  RECV_TIME_SEC = 2
  GLONASS_FREQ = 3

  PR = 4
  PR_STD = 5
  PRR = 6
  PRR_STD = 7

  SAT_POS = slice(8, 11)
  SAT_VEL = slice(11, 14)


def parse_prr(m):
  sat_pos_vel_i = np.concatenate((m[RawGNSSMeasurementIndices.SAT_POS],
                                  m[RawGNSSMeasurementIndices.SAT_VEL]))
  R_i = np.atleast_2d(m[RawGNSSMeasurementIndices.PRR_STD]**2)
  z_i = m[RawGNSSMeasurementIndices.PRR]
  return z_i, R_i, sat_pos_vel_i


def parse_pr(m):
  pseudorange = m[RawGNSSMeasurementIndices.PR]
  pseudorange_stdev = m[RawGNSSMeasurementIndices.PR_STD]
  sat_pos_freq_i = np.concatenate((m[RawGNSSMeasurementIndices.SAT_POS],
                                   np.array([m[RawGNSSMeasurementIndices.GLONASS_FREQ]])))
  z_i = np.atleast_1d(pseudorange)
  R_i = np.atleast_2d(pseudorange_stdev**2)
  return z_i, R_i, sat_pos_freq_i
