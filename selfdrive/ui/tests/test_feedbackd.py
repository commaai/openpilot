import pytest
import cereal.messaging as messaging
from cereal import car
from openpilot.common.params import Params
from openpilot.system.manager.process_config import managed_processes


@pytest.mark.skip("tmp disabled")
class TestFeedbackd:
  def setup_method(self):
    self.pm = messaging.PubMaster(['carState', 'rawAudioData'])
    self.sm = messaging.SubMaster(['audioFeedback'])

  def _send_lkas_button(self, pressed: bool):
    msg = messaging.new_message('carState')
    msg.carState.canValid = True
    msg.carState.buttonEvents = [{'type': car.CarState.ButtonEvent.Type.lkas, 'pressed': pressed}]
    self.pm.send('carState', msg)

  def _send_audio_data(self, count: int = 5):
    for _ in range(count):
      audio_msg = messaging.new_message('rawAudioData')
      audio_msg.rawAudioData.data = bytes(1600)  # 800 samples of int16
      audio_msg.rawAudioData.sampleRate = 16000
      self.pm.send('rawAudioData', audio_msg)
      self.sm.update(timeout=100)

  @pytest.mark.parametrize("record_feedback", [False, True])
  def test_audio_feedback(self, record_feedback):
    Params().put_bool("RecordAudioFeedback", record_feedback)

    managed_processes["feedbackd"].start()
    assert self.pm.wait_for_readers_to_update('carState', timeout=5)
    assert self.pm.wait_for_readers_to_update('rawAudioData', timeout=5)

    self._send_lkas_button(pressed=True)
    self._send_audio_data()
    self._send_lkas_button(pressed=False)
    self._send_audio_data()

    if record_feedback:
      assert self.sm.updated['audioFeedback'], "audioFeedback should be published when enabled"
    else:
      assert not self.sm.updated['audioFeedback'], "audioFeedback should not be published when disabled"

    self._send_lkas_button(pressed=True)
    self._send_audio_data()
    self._send_lkas_button(pressed=False)
    self._send_audio_data()

    assert not self.sm.updated['audioFeedback'], "audioFeedback should not be published after second press"

    managed_processes["feedbackd"].stop()
