import random
import numpy as np
import time
import pytest

from cereal import messaging, log, car
from openpilot.selfdrive.locationd.lagd import LateralLagEstimator, retrieve_initial_lag, masked_normalized_cross_correlation, \
                                               BLOCK_NUM_NEEDED, BLOCK_SIZE, MIN_OKAY_WINDOW_SEC
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware import PC

MAX_ERR_FRAMES = 1
DT = 0.05


def process_messages(estimator, lag_frames, n_frames, vego=20.0, rejection_threshold=0.0):
  for i in range(n_frames):
    t = i * estimator.dt
    desired_la = np.cos(10 * t) * 0.1
    actual_la = np.cos(10 * (t - lag_frames * estimator.dt)) * 0.1

    # if sample is masked out, set it to desired value (no lag)
    rejected = random.uniform(0, 1) < rejection_threshold
    if rejected:
      actual_la = desired_la

    desired_cuvature = float(desired_la / (vego ** 2))
    actual_yr = float(actual_la / vego)
    msgs = [
      (t, "carControl", car.CarControl(latActive=not rejected)),
      (t, "carState", car.CarState(vEgo=vego, steeringPressed=False)),
      (t, "controlsState", log.ControlsState(desiredCurvature=desired_cuvature)),
      (t, "livePose", log.LivePose(angularVelocityDevice=log.LivePose.XYZMeasurement(z=actual_yr, valid=True),
                                   posenetOK=True, inputsOK=True)),
      (t, "liveCalibration", log.LiveCalibrationData(rpyCalib=[0, 0, 0], calStatus=log.LiveCalibrationData.Status.calibrated)),
    ]
    for t, w, m in msgs:
      estimator.handle_log(t, w, m)
    estimator.update_points()
    estimator.update_estimate()


class TestLagd:
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = messaging.new_message('liveDelay')
    msg.liveDelay.lateralDelayEstimate = random.random()
    msg.liveDelay.validBlocks = random.randint(1, 10)
    params.put("LiveDelay", msg.to_bytes())
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes())

    saved_lag_params = retrieve_initial_lag(params, CP)
    assert saved_lag_params is not None

    lag, valid_blocks = saved_lag_params
    assert lag == msg.liveDelay.lateralDelayEstimate
    assert valid_blocks == msg.liveDelay.validBlocks

  def test_ncc(self):
    lag_frames = random.randint(1, 19)

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

    # mask out 40% of the values, and make them noise
    mask = np.random.choice([True, False], size=len(desired_sig), p=[0.6, 0.4])
    desired_sig[~mask] = np.random.normal(0, 1, size=np.sum(~mask))
    actual_sig[~mask] = np.random.normal(0, 1, size=np.sum(~mask))
    corr = masked_normalized_cross_correlation(desired_sig, actual_sig, mask, 200)[len(desired_sig) - 1:len(desired_sig) + 20]
    assert np.argmax(corr) in range(lag_frames - MAX_ERR_FRAMES, lag_frames + MAX_ERR_FRAMES + 1)

  def test_empty_estimator(self):
    mocked_CP = car.CarParams(steerActuatorDelay=0.8)
    estimator = LateralLagEstimator(mocked_CP, DT)
    msg = estimator.get_msg(True)
    assert msg.liveDelay.status == 'unestimated'
    assert np.allclose(msg.liveDelay.lateralDelay, estimator.initial_lag)
    assert np.allclose(msg.liveDelay.lateralDelayEstimate, estimator.initial_lag)
    assert msg.liveDelay.validBlocks == 0

  def test_estimator_basics(self, subtests):
    for lag_frames in range(5):
      with subtests.test(msg=f"lag_frames={lag_frames}"):
        mocked_CP = car.CarParams(steerActuatorDelay=0.8)
        estimator = LateralLagEstimator(mocked_CP, DT, min_recovery_buffer_sec=0.0, min_yr=0.0)
        process_messages(estimator, lag_frames, int(MIN_OKAY_WINDOW_SEC / DT) + BLOCK_NUM_NEEDED * BLOCK_SIZE)
        msg = estimator.get_msg(True)
        assert msg.liveDelay.status == 'estimated'
        assert np.allclose(msg.liveDelay.lateralDelay, lag_frames * DT, atol=0.01)
        assert np.allclose(msg.liveDelay.lateralDelayEstimate, lag_frames * DT, atol=0.01)
        assert np.allclose(msg.liveDelay.lateralDelayEstimateStd, 0.0, atol=0.01)
        assert msg.liveDelay.validBlocks == BLOCK_NUM_NEEDED

  def test_disabled_estimator(self):
    mocked_CP = car.CarParams(steerActuatorDelay=0.8)
    estimator = LateralLagEstimator(mocked_CP, DT, min_recovery_buffer_sec=0.0, min_yr=0.0, enabled=False)
    lag_frames = 5
    process_messages(estimator, lag_frames, int(MIN_OKAY_WINDOW_SEC / DT) + BLOCK_NUM_NEEDED * BLOCK_SIZE)
    msg = estimator.get_msg(True)
    assert msg.liveDelay.status == 'unestimated'
    assert np.allclose(msg.liveDelay.lateralDelay, 1.0, atol=0.01)
    assert np.allclose(msg.liveDelay.lateralDelayEstimate, lag_frames * DT, atol=0.01)
    assert np.allclose(msg.liveDelay.lateralDelayEstimateStd, 0.0, atol=0.01)
    assert msg.liveDelay.validBlocks == BLOCK_NUM_NEEDED

  def test_estimator_masking(self):
    mocked_CP, lag_frames = car.CarParams(steerActuatorDelay=0.8), random.randint(1, 19)
    estimator = LateralLagEstimator(mocked_CP, DT, min_recovery_buffer_sec=0.0, min_yr=0.0, min_valid_block_count=1)
    process_messages(estimator, lag_frames, (int(MIN_OKAY_WINDOW_SEC / DT) + BLOCK_SIZE) * 2, rejection_threshold=0.4)
    msg = estimator.get_msg(True)
    assert np.allclose(msg.liveDelay.lateralDelayEstimate, lag_frames * DT, atol=0.01)
    assert np.allclose(msg.liveDelay.lateralDelayEstimateStd, 0.0, atol=0.01)

  @pytest.mark.skipif(PC, reason="only on device")
  @pytest.mark.timeout(60)
  def test_estimator_performance(self):
    mocked_CP = car.CarParams(steerActuatorDelay=0.8)
    estimator = LateralLagEstimator(mocked_CP, DT)

    ds = []
    for _ in range(1000):
      st = time.perf_counter()
      estimator.update_points()
      estimator.update_estimate()
      d = time.perf_counter() - st
      ds.append(d)

    assert np.mean(ds) < DT
