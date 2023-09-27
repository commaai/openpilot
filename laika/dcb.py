from datetime import datetime
from .constants import SECS_IN_HR, SECS_IN_WEEK, \
                      SPEED_OF_LIGHT, GPS_L1, GPS_L2
from .gps_time import GPSTime
from .helpers import get_constellation
import warnings


class DCB:
  def __init__(self, prn, data):
    self.max_time_diff = 2*SECS_IN_WEEK
    self.prn = prn
    self.epoch = data['epoch']
    self.healthy = True
    if 'C1W_C2W' in data:
      self.C1W_C2W = data['C1W_C2W']
    elif 'C1P_C2P' in data:
      self.C1W_C2W = data['C1P_C2P']
    else:
      self.healthy = False
    if 'C1C_C1W' in data:
      self.C1C_C1W = data['C1C_C1W']
    elif 'C1C_C1P' in data:
      self.C1C_C1W = data['C1C_C1P']
    else:
      self.healthy = False

  def valid(self, time):
    return abs(time - self.epoch) <= self.max_time_diff and self.healthy

  def get_delay(self, signal):
    if signal == 'C1C':
      return (- SPEED_OF_LIGHT*1e-9*self.C1W_C2W*GPS_L2**2/(GPS_L1**2 - GPS_L2**2)
              + SPEED_OF_LIGHT*1e-9*self.C1C_C1W)
    if signal == 'C2P':
      return (- SPEED_OF_LIGHT*1e-9*self.C1W_C2W*GPS_L1**2/(GPS_L1**2 - GPS_L2**2))
    if signal == 'C1P':
      return (SPEED_OF_LIGHT*1e-9*self.C1C_C1W)
    ## Todo: update dcb database and get delay to include additional signals
    if signal == 'C2C':
      warnings.warn("Differential code bias not implemented for signal C2C", UserWarning)
      return 0
    if signal == 'C5C':
      warnings.warn("Differential code bias not implemented for signal C5C", UserWarning)
      return 0
    if signal == 'C6C':
      warnings.warn("Differential code bias not implemented for signal C6C", UserWarning)
      return 0
    if signal == 'C7C':
      warnings.warn("Differential code bias not implemented for signal C7C", UserWarning)
      return 0
    if signal == 'C8C':
      warnings.warn("Differential code bias not implemented for signal C8C", UserWarning)
      return 0


def parse_dcbs(file_name, SUPPORTED_CONSTELLATIONS):
  with open(file_name, 'r+') as DCB_file:
    contents = DCB_file.readlines()
  data_started = False
  dcbs_dict = {}
  for line in contents:
    if not data_started:
      if line[1:4] == 'DSB':
        data_started = True
      else:
        continue
    line_components = line.split()
    if len(line_components[2]) < 3:
      break
    prn = line_components[2]
    if get_constellation(prn) not in SUPPORTED_CONSTELLATIONS:
      continue
    dcb_type = line_components[3] + '_' + line_components[4]
    epoch = GPSTime.from_datetime(datetime.strptime(line_components[5], '%Y:%j:%f')) + 12*SECS_IN_HR
    if prn not in dcbs_dict:
      dcbs_dict[prn] = {}
    dcbs_dict[prn][dcb_type] = float(line_components[8])
    dcbs_dict[prn]['epoch'] = epoch

  dcbs = []
  for prn in dcbs_dict:
    dcbs.append(DCB(prn, dcbs_dict[prn]))
  return dcbs
