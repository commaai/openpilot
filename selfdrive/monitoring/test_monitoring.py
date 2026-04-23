import numpy as np

from cereal import log
from openpilot.common.realtime import DT_DMON
from openpilot.selfdrive.monitoring.policy import DriverMonitoring, DRIVER_MONITOR_SETTINGS

dm_settings = DRIVER_MONITOR_SETTINGS()
vision_settings = dm_settings.vision
wheeltouch_settings = dm_settings.wheeltouch

TEST_TIMESPAN = 120  # seconds
VISION_ALERT_2_INTERVAL = vision_settings._ALERT_2_INTERVAL + 1
VISION_ALERT_3_INTERVAL = vision_settings._ALERT_3_INTERVAL + 1
WHEELTOUCH_ALERT_2_INTERVAL = wheeltouch_settings._ALERT_2_INTERVAL + 1
WHEELTOUCH_ALERT_3_INTERVAL = wheeltouch_settings._ALERT_3_INTERVAL + 1

def make_msg(face_detected, distracted=False, model_uncertain=False):
  ds = log.DriverStateV2.new_message()
  ds.leftDriverData.faceOrientation = [0., 0., 0.]
  ds.leftDriverData.facePosition = [0., 0.]
  ds.leftDriverData.faceProb = 1. * face_detected
  ds.leftDriverData.eyesVisibleProb = 1.
  ds.leftDriverData.eyesClosedProb = 1. * distracted
  ds.leftDriverData.faceOrientationStd = [1.*model_uncertain, 1.*model_uncertain, 1.*model_uncertain]
  ds.leftDriverData.facePositionStd = [1.*model_uncertain, 1.*model_uncertain]
  # TODO: test both separately when e2e is used
  ds.leftDriverData.phoneProb = 0.
  return ds


# driver state from neural net, 10Hz
msg_NO_FACE_DETECTED = make_msg(False)
msg_ATTENTIVE = make_msg(True)
msg_DISTRACTED = make_msg(True, distracted=True)
msg_DISTRACTED_BUT_SOMEHOW_UNCERTAIN = make_msg(True, distracted=True, model_uncertain=vision_settings._POSE_UNCERTAINTY_THRESHOLD*1.5)

# driver interaction with car
car_interaction_DETECTED = True
car_interaction_NOT_DETECTED = False

# some common state vectors
always_no_face = [msg_NO_FACE_DETECTED] * int(TEST_TIMESPAN / DT_DMON)
always_attentive = [msg_ATTENTIVE] * int(TEST_TIMESPAN / DT_DMON)
always_distracted = [msg_DISTRACTED] * int(TEST_TIMESPAN / DT_DMON)
always_true = [True] * int(TEST_TIMESPAN / DT_DMON)
always_false = [False] * int(TEST_TIMESPAN / DT_DMON)

class TestMonitoring:
  def _run_seq(self, msgs, interaction, engaged, standstill):
    DM = DriverMonitoring()
    alert_lvls = []
    for idx in range(len(msgs)):
      DM._update_states(msgs[idx], [0, 0, 0], 0, engaged[idx], standstill[idx])
      # cal_rpy and car_speed don't matter here

      # evaluate events at 10Hz for tests
      DM._update_events(interaction[idx], engaged[idx], standstill[idx], 0)
      alert_lvls.append(DM.alert_level)
    assert len(alert_lvls) == len(msgs), f"got {len(alert_lvls)} for {len(msgs)} driverState input msgs"
    return alert_lvls, DM


  # engaged, driver is attentive all the time
  def test_fully_aware_driver(self):
    alert_lvls, d_status = self._run_seq(always_attentive, always_false, always_true, always_false)
    assert all(a == 0 for a in alert_lvls)
    assert d_status.active_policy == log.DriverMonitoringState.MonitoringPolicy.vision

  # engaged, driver is distracted and does nothing
  def test_fully_distracted_driver(self):
    alert_lvls, d_status = self._run_seq(always_distracted, always_false, always_true, always_false)
    s = d_status.vision_policy.settings
    assert alert_lvls[int(s._ALERT_1_INTERVAL / 2 / DT_DMON)] == 0
    assert alert_lvls[int((s._ALERT_1_INTERVAL + \
                    (s._ALERT_2_INTERVAL - s._ALERT_1_INTERVAL) / 2) / DT_DMON)] == 1
    assert alert_lvls[int((s._ALERT_2_INTERVAL + \
                    (s._ALERT_3_INTERVAL - s._ALERT_2_INTERVAL) / 2) / DT_DMON)] == 2
    assert alert_lvls[int((s._ALERT_3_INTERVAL + \
                    (TEST_TIMESPAN - 10 - s._ALERT_3_INTERVAL) / 2) / DT_DMON)] == 3
    assert isinstance(d_status.awareness, float)

  # engaged, no face detected the whole time, no action
  def test_fully_invisible_driver(self):
    alert_lvls, d_status = self._run_seq(always_no_face, always_false, always_true, always_false)
    s = d_status.wheeltouch_policy.settings
    assert alert_lvls[int(s._ALERT_1_INTERVAL / 2 / DT_DMON)] == 0
    assert alert_lvls[int((s._ALERT_1_INTERVAL + \
                    (s._ALERT_2_INTERVAL - s._ALERT_1_INTERVAL) / 2) / DT_DMON)] == 1
    assert alert_lvls[int((s._ALERT_2_INTERVAL + \
                    (s._ALERT_3_INTERVAL - s._ALERT_2_INTERVAL) / 2) / DT_DMON)] == 2
    assert alert_lvls[int((s._ALERT_3_INTERVAL + \
                    (TEST_TIMESPAN - 10 - s._ALERT_3_INTERVAL) / 2) / DT_DMON)] == 3
    assert d_status.active_policy == log.DriverMonitoringState.MonitoringPolicy.wheeltouch

  # engaged, down to alert level two, driver pays attention, back to normal; then back to alert level two, driver touches wheel
  #  - should have short alert level two recovery time and no alert afterwards; wheel touch only recovers when paying attention
  def test_normal_driver(self):
    ds_vector = [msg_DISTRACTED] * int(VISION_ALERT_2_INTERVAL/DT_DMON) + \
                [msg_ATTENTIVE] * int(VISION_ALERT_2_INTERVAL/DT_DMON) + \
                [msg_DISTRACTED] * int((VISION_ALERT_2_INTERVAL+2)/DT_DMON) + \
                [msg_ATTENTIVE] * (int(TEST_TIMESPAN/DT_DMON)-int((VISION_ALERT_2_INTERVAL*3+2)/DT_DMON))
    interaction_vector = [car_interaction_NOT_DETECTED] * int(VISION_ALERT_2_INTERVAL*3/DT_DMON) + \
                         [car_interaction_DETECTED] * (int(TEST_TIMESPAN/DT_DMON)-int(VISION_ALERT_2_INTERVAL*3/DT_DMON))
    alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, always_true, always_false)
    assert alert_lvls[int(VISION_ALERT_2_INTERVAL*0.5/DT_DMON)] == 0
    assert alert_lvls[int((VISION_ALERT_2_INTERVAL-0.1)/DT_DMON)] == 2
    assert alert_lvls[int(VISION_ALERT_2_INTERVAL*1.5/DT_DMON)] == 0
    assert alert_lvls[int((VISION_ALERT_2_INTERVAL*3-0.1)/DT_DMON)] == 2
    assert alert_lvls[int((VISION_ALERT_2_INTERVAL*3+0.1)/DT_DMON)] == 2
    assert alert_lvls[int((VISION_ALERT_2_INTERVAL*3+2.5)/DT_DMON)] == 0

  # engaged, down to alert level two, driver dodges camera, then comes back still distracted, down to alert level three, \
  #                          driver dodges, and then touches wheel to no avail, disengages and reengages
  #  - alert level two/three should remain after disappearance, and only disengaging clears alert level three
  def test_biggest_comma_fan(self):
    _invisible_time = 2  # seconds
    ds_vector = always_distracted[:]
    interaction_vector = always_false[:]
    op_vector = always_true[:]
    ds_vector[int(VISION_ALERT_2_INTERVAL/DT_DMON):int((VISION_ALERT_2_INTERVAL+_invisible_time)/DT_DMON)] \
                                                        = [msg_NO_FACE_DETECTED] * int(_invisible_time/DT_DMON)
    ds_vector[int((VISION_ALERT_3_INTERVAL+_invisible_time)/DT_DMON):int((VISION_ALERT_3_INTERVAL+2*_invisible_time)/DT_DMON)] \
                                                        = [msg_NO_FACE_DETECTED] * int(_invisible_time/DT_DMON)
    interaction_vector[int((VISION_ALERT_3_INTERVAL+2*_invisible_time+0.5)/DT_DMON):int((VISION_ALERT_3_INTERVAL+2*_invisible_time+1.5)/DT_DMON)] \
                                                        = [True] * int(1/DT_DMON)
    op_vector[int((VISION_ALERT_3_INTERVAL+2*_invisible_time+2.5)/DT_DMON):int((VISION_ALERT_3_INTERVAL+2*_invisible_time+3)/DT_DMON)] \
                                                        = [False] * int(0.5/DT_DMON)
    alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, op_vector, always_false)
    assert alert_lvls[int((VISION_ALERT_2_INTERVAL+0.5*_invisible_time)/DT_DMON)] == 2
    assert alert_lvls[int((VISION_ALERT_3_INTERVAL+1.5*_invisible_time)/DT_DMON)] == 3
    assert alert_lvls[int((VISION_ALERT_3_INTERVAL+2*_invisible_time+1.5)/DT_DMON)] == 3
    assert alert_lvls[int((VISION_ALERT_3_INTERVAL+2*_invisible_time+3.5)/DT_DMON)] == 0

  # engaged, invisible driver, down to alert level two, driver touches wheel; then back to alert level two, driver appears
  #  - both actions should clear the alert, but momentary appearance should not
  def test_sometimes_transparent_commuter(self):
    _visible_time = np.random.choice([0.5, 10])
    ds_vector = always_no_face[:]*2
    interaction_vector = always_false[:]*2
    ds_vector[int((2*WHEELTOUCH_ALERT_2_INTERVAL+1)/DT_DMON):int((2*WHEELTOUCH_ALERT_2_INTERVAL+1+_visible_time)/DT_DMON)] = \
                                                                                             [msg_ATTENTIVE] * int(_visible_time/DT_DMON)
    interaction_vector[int((WHEELTOUCH_ALERT_2_INTERVAL)/DT_DMON):int((WHEELTOUCH_ALERT_2_INTERVAL+1)/DT_DMON)] = [True] * int(1/DT_DMON)
    alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, 2*always_true, 2*always_false)
    assert alert_lvls[int(WHEELTOUCH_ALERT_2_INTERVAL*0.5/DT_DMON)] == 0
    assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL-0.1)/DT_DMON)] == 2
    assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL+0.1)/DT_DMON)] == 0
    if _visible_time == 0.5:
      assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL*2+1-0.1)/DT_DMON)] == 2
      assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL*2+1+0.1+_visible_time)/DT_DMON)] == 1
    elif _visible_time == 10:
      assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL*2+1-0.1)/DT_DMON)] == 2
      assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL*2+1+0.1+_visible_time)/DT_DMON)] == 0

  # engaged, invisible driver, down to alert level three, driver appears and then touches wheel, then disengages/reengages
  #  - only disengage will clear the alert
  def test_last_second_responder(self):
    _visible_time = 2  # seconds
    ds_vector = always_no_face[:]
    interaction_vector = always_false[:]
    op_vector = always_true[:]
    ds_vector[int(WHEELTOUCH_ALERT_3_INTERVAL/DT_DMON):int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time)/DT_DMON)] = \
      [msg_ATTENTIVE] * int(_visible_time/DT_DMON)
    interaction_vector[int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time)/DT_DMON):
                       int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time+1)/DT_DMON)] = [True] * int(1/DT_DMON)
    op_vector[int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time+1)/DT_DMON):
              int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time+0.5)/DT_DMON)] = [False] * int(0.5/DT_DMON)
    alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, op_vector, always_false)
    assert alert_lvls[int(WHEELTOUCH_ALERT_2_INTERVAL*0.5/DT_DMON)] == 0
    assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL-0.1)/DT_DMON)] == 2
    assert alert_lvls[int((WHEELTOUCH_ALERT_3_INTERVAL-0.1)/DT_DMON)] == 3
    assert alert_lvls[int((WHEELTOUCH_ALERT_3_INTERVAL+0.5*_visible_time)/DT_DMON)] == 3
    assert alert_lvls[int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time+0.5)/DT_DMON)] == 3
    assert alert_lvls[int((WHEELTOUCH_ALERT_3_INTERVAL+_visible_time+1+0.1)/DT_DMON)] == 0

  # disengaged, always distracted driver
  #  - dm should stay quiet when not engaged
  def test_pure_dashcam_user(self):
    alert_lvls, _ = self._run_seq(always_distracted, always_false, always_false, always_false)
    assert all(a == 0 for a in alert_lvls)

  # engaged, car stops at traffic light, down to alert level two, no action, then car starts moving
  #  - should only reach green when stopped, but continues counting down on launch
  def test_long_traffic_light_victim(self):
    _redlight_time = 60  # seconds
    standstill_vector = always_true[:]
    standstill_vector[int(_redlight_time/DT_DMON):] = [False] * int((TEST_TIMESPAN-_redlight_time)/DT_DMON)
    alert_lvls, d_status = self._run_seq(always_distracted, always_false, always_true, standstill_vector)
    s = d_status.vision_policy.settings
    assert alert_lvls[int((_redlight_time-0.1)/DT_DMON)] == 0
    _vision_alert_1_to_2_interval = s._ALERT_2_INTERVAL - s._ALERT_1_INTERVAL
    assert alert_lvls[int((_redlight_time+0.5)/DT_DMON)] == 1
    assert alert_lvls[int((_redlight_time+_vision_alert_1_to_2_interval+0.5)/DT_DMON)] == 2

  # engaged, distracted while moving, then car stops after reaching alert level two
  #  - should reset timer to pre green at standstill
  def test_distracted_then_stops(self):
    _stop_time = VISION_ALERT_2_INTERVAL + 1  # stop 1 second after reaching alert level two
    standstill_vector = always_false[:]
    standstill_vector[int(_stop_time/DT_DMON):] = [True] * int((TEST_TIMESPAN-_stop_time)/DT_DMON)
    alert_lvls, _ = self._run_seq(always_distracted, always_false, always_true, standstill_vector)
    # just before and briefly after stopping: alert level two; goes away quickly after stopped
    assert alert_lvls[int((_stop_time+0.1)/DT_DMON)] == 2
    assert alert_lvls[int((_stop_time+0.5)/DT_DMON)] == 0

  # engaged, model is somehow uncertain and driver is distracted
  #  - should fall back to wheel touch after uncertain alert
  def test_somehow_indecisive_model(self):
    ds_vector = [msg_DISTRACTED_BUT_SOMEHOW_UNCERTAIN] * int(TEST_TIMESPAN/DT_DMON)
    interaction_vector = always_false[:]
    alert_lvls, d_status = self._run_seq(ds_vector, interaction_vector, always_true, always_false)
    s = d_status.vision_policy.settings
    assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL-1+DT_DMON*s._WHEELTOUCH_FALLBACK_TIME-0.1)/DT_DMON)] == 1
    assert alert_lvls[int((WHEELTOUCH_ALERT_2_INTERVAL-1+DT_DMON*s._WHEELTOUCH_FALLBACK_TIME+0.1)/DT_DMON)] == 2
    assert alert_lvls[int((WHEELTOUCH_ALERT_3_INTERVAL-1+DT_DMON*s._WHEELTOUCH_FALLBACK_TIME+0.1)/DT_DMON)] == 3
