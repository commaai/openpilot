from openpilot.selfdrive.locationd.locationd import LocationEstimator


def test_get_msg_uninitialized_filter_time():
  msg = LocationEstimator(False).get_msg(sensors_valid=True, inputs_valid=True, filter_initialized=False)

  assert not msg.valid
  assert msg.livePose.timestamp == 0
