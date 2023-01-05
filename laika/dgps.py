import os
import numpy as np
from datetime import datetime

from .gps_time import GPSTime
from .constants import SECS_IN_YEAR
from . import raw_gnss as raw
from . import opt
from .rinex_file import RINEXFile
from .downloader import download_cors_coords
from .helpers import get_constellation


def mean_filter(delay):
  d2 = delay.copy()
  max_step = 10
  for i in range(max_step, len(delay) - max_step):
    finite_idxs = np.where(np.isfinite(delay[i - max_step:i + max_step]))
    if max_step in finite_idxs[0]:
      step = min([max_step, finite_idxs[0][-1] - max_step, max_step - finite_idxs[0][0]])
      d2[i] = np.nanmean(delay[i - step:i + step + 1])
  return d2


def download_and_parse_station_postions(cors_station_positions_path, cache_dir):
  if not os.path.isfile(cors_station_positions_path):
    cors_stations = {}
    coord_file_paths = download_cors_coords(cache_dir=cache_dir)
    for coord_file_path in coord_file_paths:
      try:
        station_id = coord_file_path.split('/')[-1][:4]
        with open(coord_file_path, 'r+') as coord_file:
          contents = coord_file.readlines()
        phase_center = False
        for line_number in range(len(contents)):
          if 'L1 Phase Center' in contents[line_number]:
            phase_center = True
          if not phase_center and 'ITRF2014 POSITION' in contents[line_number]:
            velocity = [float(contents[line_number+8].split()[3]),
                        float(contents[line_number+9].split()[3]),
                        float(contents[line_number+10].split()[3])]
          if phase_center and 'ITRF2014 POSITION' in contents[line_number]:
            epoch = GPSTime.from_datetime(datetime(2005,1,1))
            position = [float(contents[line_number+2].split()[3]),
                        float(contents[line_number+3].split()[3]),
                        float(contents[line_number+4].split()[3])]
            cors_stations[station_id] = [epoch, position, velocity]
            break
      except ValueError:
        pass
    cors_station_positions_file = open(cors_station_positions_path, 'wb')
    np.save(cors_station_positions_file, cors_stations)
    cors_station_positions_file.close()


def get_closest_station_names(pos, k=5, max_distance=100000, cache_dir='/tmp/gnss/'):
  from scipy.spatial import cKDTree

  cors_station_positions_dict = load_cors_station_positions(cache_dir)
  station_ids = list(cors_station_positions_dict.keys())
  station_positions = []
  for station_id in station_ids:
    station_positions.append(cors_station_positions_dict[station_id][1])
  tree = cKDTree(station_positions)
  distances, idxs = tree.query(pos, k=k, distance_upper_bound=max_distance)
  return np.array(station_ids)[idxs]


def load_cors_station_positions(cache_dir):
  cors_station_positions_path = cache_dir + 'cors_coord/cors_station_positions'
  download_and_parse_station_postions(cors_station_positions_path, cache_dir)
  with open(cors_station_positions_path, 'rb') as f:
    return np.load(f, allow_pickle=True).item()  # pylint: disable=unexpected-keyword-arg


def get_station_position(station_id, cache_dir='/tmp/gnss/', time=GPSTime.from_datetime(datetime.utcnow())):
  cors_station_positions_dict = load_cors_station_positions(cache_dir)
  epoch, pos, vel = cors_station_positions_dict[station_id]
  return ((time - epoch)/SECS_IN_YEAR)*np.array(vel) + np.array(pos)


def parse_dgps(station_id, station_obs_file_path, dog, max_distance=100000, required_constellations=['GPS']):
  station_pos = get_station_position(station_id, cache_dir=dog.cache_dir)
  obsdata = RINEXFile(station_obs_file_path)
  measurements = raw.read_rinex_obs(obsdata)

  # if not all constellations in first 100 epochs bail
  detected_constellations = set()
  for m in sum(measurements[:100],[]):
    detected_constellations.add(get_constellation(m.prn))
  for constellation in required_constellations:
    if constellation not in detected_constellations:
      return None

  proc_measurements = []
  for measurement in measurements:
    proc_measurements.append(raw.process_measurements(measurement, dog=dog))
  # sample at 30s
  if len(proc_measurements) > 2880:
    proc_measurements = proc_measurements[::int(len(proc_measurements)/2880)]
  if len(proc_measurements) != 2880:
    return None

  station_delays = {}
  n = len(proc_measurements)
  for signal in ['C1C', 'C2P']:
    times = []
    station_delays[signal] = {}
    for i, proc_measurement in enumerate(proc_measurements):
      times.append(proc_measurement[0].recv_time)
      Fx_pos = opt.pr_residual(proc_measurement, signal=signal)
      residual, _ = Fx_pos(list(station_pos) + [0,0])
      residual = -np.array(residual)
      for j, m in enumerate(proc_measurement):
        prn = m.prn
        if prn not in station_delays[signal]:
          station_delays[signal][prn] = np.nan*np.ones(n)
        station_delays[signal][prn][i] = residual[j]
  assert len(times) == n

  # TODO crude way to get dgps station's clock errors,
  # could this be biased? Only use GPS for convenience.
  model_delays = {}
  for prn in station_delays['C1C']:
    if get_constellation(prn) == 'GPS':
      model_delays[prn] = np.nan*np.zeros(n)
      for i in range(n):
        model_delays[prn][i] = dog.get_delay(prn, times[i], station_pos, no_dgps=True)
  station_clock_errs = np.zeros(n)
  for i in range(n):
    station_clock_errs[i] = np.nanmean([(station_delays['C1C'][prn][i] - model_delays[prn][i]) for prn in model_delays])

  # remove clock errors and smooth out signal
  for prn in station_delays['C1C']:
    station_delays['C1C'][prn] = mean_filter(station_delays['C1C'][prn] - station_clock_errs)
  for prn in station_delays['C2P']:
    station_delays['C2P'][prn] = station_delays['C2P'][prn] - station_clock_errs

  return DGPSDelay(station_id, station_pos, station_delays,
                   times, max_distance)


class DGPSDelay:
  def __init__(self, station_id, station_pos,
               station_delays, station_delays_t, max_distance):
    self.id = station_id
    self.pos = station_pos
    self.delays = station_delays
    self.delays_t = station_delays_t
    self.max_distance = max_distance

  def get_delay(self, prn, time, signal='C1C'):
    time_index = int((time - self.delays_t[0])/30)
    assert abs(self.delays_t[time_index] - time) < 30
    if prn in self.delays[signal] and np.isfinite(self.delays[signal][prn][time_index]):
      return self.delays[signal][prn][time_index]
    return None

  def valid(self, time, recv_pos):
    return (np.linalg.norm(recv_pos - self.pos) <= self.max_distance and
            time - self.delays_t[0] > -30 and
            self.delays_t[-1] - time > -30)
