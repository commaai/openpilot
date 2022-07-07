import numpy as np
from laika.raw_gnss import GNSSMeasurement

def parse_prr(m):
  sat_pos_vel_i = np.concatenate((m[GNSSMeasurement.SAT_POS],
                                  m[GNSSMeasurement.SAT_VEL]))
  R_i = np.atleast_2d(m[GNSSMeasurement.PRR_STD]**2)
  z_i = m[GNSSMeasurement.PRR]
  return z_i, R_i, sat_pos_vel_i

def parse_pr(m):
  pseudorange = m[GNSSMeasurement.PR]
  pseudorange_stdev = m[GNSSMeasurement.PR_STD]
  sat_pos_freq_i = np.concatenate((m[GNSSMeasurement.SAT_POS],
                                   np.array([m[GNSSMeasurement.GLONASS_FREQ]])))
  z_i = np.atleast_1d(pseudorange)
  R_i = np.atleast_2d(pseudorange_stdev**2)
  return z_i, R_i, sat_pos_freq_i

