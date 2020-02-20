import numpy as np
import os
from bisect import bisect
from tqdm import tqdm
from cffi import FFI

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'generated'))


def write_code(name, code, header):
  if not os.path.exists(GENERATED_DIR):
    os.mkdir(GENERATED_DIR)

  open(os.path.join(GENERATED_DIR, f"{name}.cpp"), 'w').write(code)
  open(os.path.join(GENERATED_DIR, f"{name}.h"), 'w').write(header)


def load_code(name):
  shared_fn = os.path.join(GENERATED_DIR, f"lib{name}.so")
  header_fn = os.path.join(GENERATED_DIR, f"{name}.h")
  header = open(header_fn).read()

  ffi = FFI()
  ffi.cdef(header)
  return (ffi, ffi.dlopen(shared_fn))


class KalmanError(Exception):
  pass


class ObservationKind():
  UNKNOWN = 0
  NO_OBSERVATION = 1
  GPS_NED = 2
  ODOMETRIC_SPEED = 3
  PHONE_GYRO = 4
  GPS_VEL = 5
  PSEUDORANGE_GPS = 6
  PSEUDORANGE_RATE_GPS = 7
  SPEED = 8
  NO_ROT = 9
  PHONE_ACCEL = 10
  ORB_POINT = 11
  ECEF_POS = 12
  CAMERA_ODO_TRANSLATION = 13
  CAMERA_ODO_ROTATION = 14
  ORB_FEATURES = 15
  MSCKF_TEST = 16
  FEATURE_TRACK_TEST = 17
  LANE_PT = 18
  IMU_FRAME = 19
  PSEUDORANGE_GLONASS = 20
  PSEUDORANGE_RATE_GLONASS = 21
  PSEUDORANGE = 22
  PSEUDORANGE_RATE = 23

  CAL_DEVICE_FRAME_XY_SPEED = 24  # (x, y) [m/s]
  CAL_DEVICE_FRAME_YAW_RATE = 25  # [rad/s]
  STEER_ANGLE = 26  # [rad]
  ANGLE_OFFSET_FAST = 27  # [rad]

  names = [
    'Unknown',
    'No observation',
    'GPS NED',
    'Odometric speed',
    'Phone gyro',
    'GPS velocity',
    'GPS pseudorange',
    'GPS pseudorange rate',
    'Speed',
    'No rotation',
    'Phone acceleration',
    'ORB point',
    'ECEF pos',
    'camera odometric translation',
    'camera odometric rotation',
    'ORB features',
    'MSCKF test',
    'Feature track test',
    'Lane ecef point',
    'imu frame eulers',
    'GLONASS pseudorange',
    'GLONASS pseudorange rate',

    'Calibrated Device Frame x,y speed',
    'Calibrated Device Frame yaw rate',
    'Steer Angle',
    'Fast Angle Offset',
  ]

  @classmethod
  def to_string(cls, kind):
    return cls.names[kind]



SAT_OBS = [ObservationKind.PSEUDORANGE_GPS,
           ObservationKind.PSEUDORANGE_RATE_GPS,
           ObservationKind.PSEUDORANGE_GLONASS,
           ObservationKind.PSEUDORANGE_RATE_GLONASS]


def run_car_ekf_offline(kf, observations_by_kind):
  from laika.raw_gnss import GNSSMeasurement
  observations = []
  # create list of observations with element format: [kind, time, data]
  for kind in observations_by_kind:
    for t, data in zip(observations_by_kind[kind][0], observations_by_kind[kind][1]):
      observations.append([t, kind, data])
  observations.sort(key=lambda obs: obs[0])

  times, estimates = run_observations_through_filter(kf, observations)

  forward_states = np.stack(e[1] for e in estimates)
  forward_covs = np.stack(e[3] for e in estimates)
  smoothed_states, smoothed_covs = kf.rts_smooth(estimates)

  observations_dict = {}
  # TODO assuming observations and estimates
  # are same length may not work with VO
  for e in estimates:
    t = e[4]
    kind = str(int(e[5]))
    res = e[6]
    z = e[7]
    ea = e[8]
    if len(z) == 0:
      continue
    if kind not in observations_dict:
      observations_dict[kind] = {}
      observations_dict[kind]['t'] = np.array(len(z)*[t])
      observations_dict[kind]['z'] = np.array(z)
      observations_dict[kind]['ea'] = np.array(ea)
      observations_dict[kind]['residual'] = np.array(res)
    else:
      observations_dict[kind]['t'] = np.append(observations_dict[kind]['t'], np.array(len(z)*[t]))
      observations_dict[kind]['z'] = np.vstack((observations_dict[kind]['z'], np.array(z)))
      observations_dict[kind]['ea'] = np.vstack((observations_dict[kind]['ea'], np.array(ea)))
      observations_dict[kind]['residual'] = np.vstack((observations_dict[kind]['residual'], np.array(res)))

  # add svIds to gnss data
  for kind in map(str, SAT_OBS):
    if int(kind) in observations_by_kind and kind in observations_dict:
      observations_dict[kind]['svIds'] = np.array([])
      observations_dict[kind]['CNO'] = np.array([])
      observations_dict[kind]['std'] = np.array([])
      for obs in observations_by_kind[int(kind)][1]:
        observations_dict[kind]['svIds'] = np.append(observations_dict[kind]['svIds'],
                                                     np.array([obs[:,GNSSMeasurement.PRN]]))
        observations_dict[kind]['std'] = np.append(observations_dict[kind]['std'],
                                                   np.array([obs[:,GNSSMeasurement.PR_STD]]))
  return smoothed_states, smoothed_covs, forward_states, forward_covs, times, observations_dict


def run_observations_through_filter(kf, observations, filter_time=None):
  estimates = []

  for obs in tqdm(observations):
    t = obs[0]
    kind = obs[1]
    data = obs[2]
    estimates.append(kf.predict_and_observe(t, kind, data))
  times = [x[4] for x in estimates]
  return times, estimates


def save_residuals_plot(obs, save_path, data_name):
  import matplotlib.pyplot as plt
  import mpld3  # pylint: disable=import-error
  fig = plt.figure(figsize=(10,20))
  fig.suptitle('Residuals of ' + data_name, fontsize=24)
  n = len(list(obs.keys()))
  start_times = [obs[kind]['t'][0] for kind in obs]
  start_time = min(start_times)
  xlims = [start_time + 3, start_time + 60]

  for i, kind in enumerate(obs):
    ax = fig.add_subplot(n, 1, i+1)
    ax.set_xlim(xlims)
    t = obs[kind]['t']
    res = obs[kind]['residual']
    start_idx = bisect(t, xlims[0])
    if len(res) == start_idx:
      continue
    ylim = max(np.linalg.norm(res[start_idx:], axis=1))
    ax.set_ylim([-ylim, ylim])
    if int(kind) in SAT_OBS:
      svIds = obs[kind]['svIds']
      for svId in set(svIds):
        svId_idx = (svIds == svId)
        t = obs[kind]['t'][svId_idx]
        res = obs[kind]['residual'][svId_idx]
        ax.plot(t, res, label='SV ' + str(int(svId)))
        ax.legend(loc='right')
    else:
      ax.plot(t, res)
    plt.title('Residual of kind ' + ObservationKind.to_string(int(kind)), fontsize=20)
  plt.tight_layout()
  os.makedirs(save_path)
  mpld3.save_html(fig, save_path + 'residuals_plot.html')
