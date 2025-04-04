import random
import numpy as np
import time
from unittest import mock
import pytest

from cereal import messaging
from openpilot.selfdrive.locationd.lagd import LateralLagEstimator, retrieve_initial_lag, masked_normalized_cross_correlation
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware import PC

MAX_ERR_FRAMES = 1


class TestLagd:
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = messaging.new_message('liveDelay')
    msg.liveDelay.lateralDelayEstimate = random.random()
    msg.liveDelay.validBlocks = random.randint(1, 10)
    params.put("LiveLag", msg.to_bytes())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())

    saved_lag_params = retrieve_initial_lag(params, CP)
    assert saved_lag_params is not None

    lag, valid_blocks = saved_lag_params
    assert lag == msg.liveDelay.lateralDelayEstimate
    assert valid_blocks == msg.liveDelay.validBlocks

  def test_ncc(self):
    lag_frames = random.randint(1, 20)

    desired_sig = np.sin(np.arange(0.0, 10.0, 0.1))
    actual_sig = np.sin(np.arange(0.0, 10.0, 0.1) - lag_frames * 0.1)
    mask = np.ones(len(desired_sig), dtype=bool)

    corr = masked_normalized_cross_correlation(desired_sig, actual_sig, mask, 200)[len(desired_sig) - 1:len(desired_sig) + 20]
    assert np.argmax(corr) == lag_frames

    # add some noise
    desired_sig += np.random.normal(0, 0.05, len(desired_sig))
    actual_sig += np.random.normal(0, 0.05, len(actual_sig))
    corr = masked_normalized_cross_correlation(desired_sig, actual_sig, mask, 200)[len(desired_sig) - 1:len(desired_sig) + 20]
    assert np.argmax(corr)  in range(lag_frames - MAX_ERR_FRAMES, lag_frames + MAX_ERR_FRAMES + 1)

  def test_estimator(self):
    class ZeroMock(mock.Mock):
      def __getattr__(self, *args):
        return 0

    dt = 0.05
    lag_frames = random.randint(1, 20)

    mocked_CP = mock.Mock(steerActuatorDelay=1.0)
    estimator = LateralLagEstimator(mocked_CP, 0.05,
                                    block_count=10, min_valid_block_count=0,
                                    block_size=1, okay_window_sec=100 * dt,
                                    min_recovery_buffer_sec=0, min_yr=0)
    for i in range(100):
      t = i * dt
      vego = 20.0
      desired_cuvature = np.cos(t) * 100 / (vego ** 2)
      actual_yr = np.cos(t - lag_frames * dt) * 100 / vego
      msgs = [
        (t, "carControl", mock.Mock(latActive=True)),
        (t, "carState", mock.Mock(vEgo=vego, steeringPressed=False)),
        (t, "controlsState", mock.Mock(desiredCurvature=desired_cuvature,
                                       lateralControlState=mock.Mock(which=mock.Mock(return_value='debugControlState'), debugControlState=ZeroMock()))),
        (t, "livePose", mock.Mock(orientationNED=ZeroMock(),
                                  velocityDevice=ZeroMock(),
                                  accelerationDevice=ZeroMock(),
                                  angularVelocityDevice=ZeroMock(z=actual_yr))),
      ]
      for t, w, m in msgs:
        estimator.handle_log(t, w, m)
      estimator.update_points()
    estimator.update_estimate()

    # expect one block filled, with lateralDelayEstimate equal to lateralDelay equal to lag_frames
    output = estimator.get_msg(True)
    assert np.allclose(output.liveDelay.lateralDelay, lag_frames * dt, atol=0.01)
    assert output.liveDelay.status == 'estimated'
    assert output.liveDelay.validBlocks == 1

  @pytest.mark.skipif(PC, reason="only on device")
  def test_estimator_performance(self):
    mocked_CP = mock.Mock(steerActuatorDelay=0.1)
    estimator = LateralLagEstimator(mocked_CP, 0.05)

    ds = []
    test_start = time.perf_counter()
    for _ in range(1000):
      st = time.perf_counter()
      estimator.update_points()
      estimator.update_estimate()
      d = time.perf_counter() - st
      ds.append(d)

      # limit the test to 20 seconds
      if time.perf_counter() - test_start > 20.0:
        break

    assert np.mean(ds) < 0.05
