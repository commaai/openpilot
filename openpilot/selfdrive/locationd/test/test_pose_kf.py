import numpy as np

from openpilot.selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from openpilot.selfdrive.locationd.models.pose_kf import PoseKalman, States


def test_accelerometer_temperature_compensation():
  kf = PoseKalman(GENERATED_DIR, max_rewind_age=0.8)
  state = PoseKalman.initial_x.copy()
  state[States.ACCEL_TEMP_COEFF] = [0.01, 0.0, 0.0]
  kf.init_state(state, covs=np.eye(len(state)) * 1e-6, filter_time=0.0)

  kf.set_temperature_delta(10.0)
  result = kf.predict_and_observe(0.01, ObservationKind.PHONE_ACCEL, [0.1, 0.0, -9.81])

  assert result is not None
  assert np.allclose(result[6][0], 0.0)
