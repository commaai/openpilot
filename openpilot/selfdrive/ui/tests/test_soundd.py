import threading

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log, messaging
from openpilot.cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SELFDRIVE_STATE_TIMEOUT, check_selfdrive_timeout_alert

AudibleAlert = log.SelfdriveState.AudibleAlert


class TestSoundd(OpenpilotTestCase):
  def test_check_selfdrive_timeout_alert(self, mocker):
    sm = SubMaster(['selfdriveState'])
    pm = PubMaster(['selfdriveState'])

    cs = messaging.new_message('selfdriveState')
    cs.selfdriveState.enabled = True
    threading.Timer(0.01, pm.send, args=("selfdriveState", cs)).start()
    sm.update(100)
    assert sm.updated['selfdriveState']

    sm.recv_time['selfdriveState'] = 0
    clock = mocker.patch("openpilot.selfdrive.ui.soundd.time.monotonic", return_value=SELFDRIVE_STATE_TIMEOUT)
    assert not check_selfdrive_timeout_alert(sm)

    clock.return_value = SELFDRIVE_STATE_TIMEOUT + 0.1
    assert check_selfdrive_timeout_alert(sm)

    clock.return_value = SELFDRIVE_STATE_TIMEOUT + 10
    assert not check_selfdrive_timeout_alert(sm)

  # TODO: add test with micd for checking that soundd actually outputs sounds


class TestLivestreamPlayback:
  @staticmethod
  def message(value=8192, frames=1920):
    import numpy as np
    msg = messaging.new_message('livestreamAudio', valid=True)
    msg.livestreamAudio.sampleRate = 48000
    msg.livestreamAudio.data = np.full(frames, value, dtype=np.int16).tobytes()
    return msg

  def test_playback_expires_and_flushes(self, mocker):
    import numpy as np
    from openpilot.selfdrive.ui.soundd import LivestreamPlayback
    playback = LivestreamPlayback()
    msg = self.message()
    playback.enqueue(msg)
    np.testing.assert_allclose(playback.render(480), 0.25)
    empty = self.message(frames=0)
    playback.enqueue(empty)
    assert not playback.render(480).any()
    playback.enqueue(msg)
    mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic_ns', return_value=msg.logMonoTime + 300_000_000)
    assert not playback.render(960).any()

  def test_backlog_is_bounded(self):
    import numpy as np
    from openpilot.selfdrive.ui.soundd import LivestreamPlayback
    playback = LivestreamPlayback()
    for _ in range(10):
      playback.enqueue(self.message(value=4096))
    playback.eq.reset()
    for _ in range(6):
      playback.enqueue(self.message(value=8192))
    np.testing.assert_allclose(playback.render(960 * 6), 0.25)
    assert not playback.render(960).any()

  def test_alerts_take_priority_over_voice(self, mocker):
    import numpy as np
    from openpilot.selfdrive.ui.soundd import Soundd
    mocker.patch.object(Soundd, 'load_sounds')
    sound = Soundd()
    sound.loaded_sounds = {AudibleAlert.engage: np.full(48000, 0.5, dtype=np.float32)}
    sound.current_volume = 1.0
    output = np.empty((960, 1), dtype=np.float32)
    sound.usb_stream = mocker.Mock()
    sound.livestream.enqueue(self.message())
    sound.callback(output, 960, None, None)
    np.testing.assert_allclose(output, 0)  # USB owns speech while connected
    usb_output = np.empty((960, 2), dtype=np.float32)
    sound.usb_callback(usb_output, 960, None, None)
    np.testing.assert_allclose(usb_output, 0.25)
    sound.current_alert = AudibleAlert.engage
    sound.livestream.enqueue(self.message())
    sound.callback(output, 960, None, None)
    np.testing.assert_allclose(output, 0.5)
    sound.usb_callback(usb_output, 960, None, None)
    np.testing.assert_allclose(usb_output, 0)


  def test_voice_preserves_level_and_limits_peaks(self):
    import numpy as np
    from openpilot.selfdrive.ui.soundd import LivestreamPlayback
    playback = LivestreamPlayback()
    for value in (0, 128, -128, 32767, -32768):
      playback.clear()
      playback.enqueue(self.message(value=value))
      output = playback.render(960)
      assert np.isfinite(output).all()
      assert np.max(np.abs(output)) <= 1.0
      if abs(value) == 128:
        np.testing.assert_allclose(output, value / 32768, rtol=0.001)
      elif value:
        assert np.all(np.sign(output) == np.sign(value))
      else:
        assert not output.any()


  def test_usb_selection_and_failure_does_not_affect_alerts(self, mocker):
    from openpilot.selfdrive.ui.soundd import Soundd
    mocker.patch.object(Soundd, 'load_sounds')
    clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=10)
    sound = Soundd()
    discover = mocker.patch('openpilot.selfdrive.ui.soundd.find_usb_output', return_value=None)
    factory = mocker.patch('openpilot.selfdrive.ui.soundd.USBSpeaker')
    sound.update_usb_stream()
    factory.assert_not_called()
    # A speaker connected after startup must be discovered on the next retry.
    discover.return_value = 'plughw:2,0'
    clock.return_value += 3
    factory.return_value.start.side_effect = RuntimeError('unplugged')
    sound.update_usb_stream()
    assert sound.usb_stream is None
    factory.return_value.close.assert_called_once()
    sound.update_usb_stream()
    assert factory.call_count == 1
    clock.return_value += 3
    factory.return_value.start.side_effect = None
    sound.update_usb_stream()
    assert sound.usb_stream is factory.return_value
    assert factory.call_args.args[0] == 'plughw:2,0'
    # A disappeared device must be closed even if the backend still says active.
    factory.return_value.active = True
    discover.return_value = None
    sound.update_usb_stream()
    assert sound.usb_stream is None
    assert factory.return_value.close.call_count == 2

  def test_missing_usb_uses_builtin_with_same_gain(self, mocker):
    import numpy as np
    from openpilot.selfdrive.ui.soundd import Soundd
    mocker.patch.object(Soundd, 'load_sounds')
    sound = Soundd()
    mocker.patch('openpilot.selfdrive.ui.soundd.find_usb_output', return_value=None)
    factory = mocker.patch('openpilot.selfdrive.ui.soundd.USBSpeaker')
    sound.update_usb_stream()
    factory.assert_not_called()
    sound.livestream.enqueue(self.message())
    output = np.empty((960, 1), dtype=np.float32)
    sound.callback(output, 960, None, None)
    np.testing.assert_allclose(output, 0.25)
    sound.usb_stream = mocker.Mock(active=False)
    sound.update_usb_stream()
    assert sound.usb_stream is None
    sound.livestream.enqueue(self.message())
    sound.callback(output, 960, None, None)
    np.testing.assert_allclose(output, 0.25)


def test_speaker_test_tone_is_bounded_and_alerts_still_override(mocker):
  import numpy as np
  from openpilot.selfdrive.ui.soundd import Soundd
  mocker.patch.object(Soundd, 'load_sounds')
  sound = Soundd()
  sound.usb_stream = mocker.Mock()
  output = np.empty((960, 2), dtype=np.float32)
  sound.request_test_sound()
  sound.usb_callback(output, 960, None, None)
  assert 0 < np.max(np.abs(output)) <= 0.12
  sound.current_alert = AudibleAlert.warningImmediate
  sound.usb_callback(output, 960, None, None)
  assert not output.any()
  for _ in range(148):
    sound.usb_callback(output, 960, None, None)
  assert sound.test_tone_frame is None



def test_speech_buffer_bridges_short_packet_jitter_and_reprimes(mocker):
  import numpy as np
  from openpilot.selfdrive.ui.soundd import LivestreamPlayback
  playback = LivestreamPlayback()
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic_ns', return_value=1_000_000_000)
  def feed():
    msg = TestLivestreamPlayback.message(frames=960)
    msg.logMonoTime = clock.return_value
    playback.enqueue(msg)
  feed()
  assert not playback.render(960).any()  # wait for the small startup buffer
  clock.return_value += 20_000_000
  feed()
  np.testing.assert_allclose(playback.render(960), 0.25)
  # Next packet is late; the reserved packet bridges that interval.
  clock.return_value += 20_000_000
  np.testing.assert_allclose(playback.render(960), 0.25)
  clock.return_value += 10_000_000
  feed()
  np.testing.assert_allclose(playback.render(960), 0.25)
  assert not playback.render(960).any()  # actual starvation requires rebuffering
  feed()
  assert not playback.render(960).any()
  feed()
  np.testing.assert_allclose(playback.render(960), 0.25)
  playback.clear()
  assert not playback.render(960).any()
