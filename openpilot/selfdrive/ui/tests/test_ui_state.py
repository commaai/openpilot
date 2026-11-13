from openpilot.cereal import messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.ui_state import ChestnutState, UIState


class TestChestnutState(OpenpilotTestCase):
  def setUp(self):
    # Exercise the state update without starting the UI singleton's workers.
    self.ui = object.__new__(UIState)
    self.ui.sm = messaging.SubMaster(["deviceState", "modelV2"])
    self.ui.sm.simulation = False
    self.ui.started = True
    self.ui.started_frame = 0
    self.ui.chestnut_present = True
    self.ui.chestnut_compiled = True
    self.ui.chestnut_active = True
    self.ui.chestnut_loading = False
    self.ui.chestnut_alert = False
    self.ui.chestnut_state = ChestnutState.LOADING
    self.now = 1.0
    self.publish()
    self.publish()

  def publish(self, *, detected=True, big=True, valid=True, model=True, dt=0.05):
    self.now += dt
    device_msg = messaging.new_message("deviceState", valid=True)
    device_msg.deviceState.chestnutPresent = detected
    msgs = [device_msg.as_reader()]
    if model:
      model_msg = messaging.new_message("modelV2", valid=valid)
      model_msg.modelV2.big = big
      msgs.append(model_msg.as_reader())
    self.ui.sm.update_msgs(self.now, msgs)

  def test_failed_recovers_when_valid_big_output_resumes(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.ACTIVE)

  def test_loading_gap_does_not_latch_failure_after_first_big_frame(self):
    self.ui.chestnut_loading = True
    self.publish(big=False)
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.LOADING)

    self.ui.chestnut_loading = False
    self.publish(model=False, dt=1.0)
    self.assertFalse(self.ui.sm.alive["modelV2"])
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

    self.publish()
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.ACTIVE)

  def test_failed_does_not_recover_with_stale_output(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.publish(model=False, dt=1.0)
    self.assertFalse(self.ui.sm.alive["modelV2"])
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

  def test_failed_does_not_recover_with_invalid_output(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.publish(valid=False)
    self.assertTrue(self.ui.sm.alive["modelV2"])
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

  def test_failed_does_not_recover_with_small_model(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.publish(big=False)
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

  def test_failed_does_not_recover_without_active_confirmation(self):
    for active in (None, False):
      with self.subTest(active=active):
        self.ui.chestnut_state = ChestnutState.FAILED
        self.ui.chestnut_active = active
        self.ui._update_chestnut_state()
        self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

  def test_failed_does_not_recover_while_device_is_not_detected(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.publish(detected=False)
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

  def test_loading_takes_precedence_over_healthy_output(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.ui.chestnut_loading = True
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.LOADING)

  def test_previous_drive_frame_does_not_confirm_active(self):
    self.ui.started_frame = self.ui.sm.recv_frame["modelV2"]
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.LOADING)

    self.publish()
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.ACTIVE)

  def test_explicit_failure_after_active_is_still_reported(self):
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.ACTIVE)

    self.ui.chestnut_active = False
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)

  def test_disconnected_and_uncompiled_states_are_preserved(self):
    self.ui.chestnut_state = ChestnutState.FAILED
    self.ui.chestnut_present = False
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.DISCONNECTED)

    self.ui.chestnut_present = True
    self.ui.chestnut_compiled = False
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.UNCOMPILED)

  def test_offroad_state_and_alert_are_preserved(self):
    self.ui.started = False
    self.ui.chestnut_state = ChestnutState.FAILED
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.READY)

    self.ui.chestnut_alert = True
    self.ui._update_chestnut_state()
    self.assertEqual(self.ui.chestnut_state, ChestnutState.FAILED)
