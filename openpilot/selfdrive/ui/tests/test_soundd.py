import threading
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log, messaging
from openpilot.cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SAMPLE_RATE, SELFDRIVE_STATE_TIMEOUT, WEBRTC_MAX_BUFFER, Soundd, check_selfdrive_timeout_alert

AudibleAlert = log.SelfdriveState.AudibleAlert


class TestSoundd(OpenpilotTestCase):
  def test_webrtc_audio_buffer(self, mocker):
    mocker.patch.object(Soundd, "load_sounds")
    soundd = Soundd()
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    soundd.add_webrtc_audio(samples.tobytes(), SAMPLE_RATE)
    soundd.webrtc_playing = True

    np.testing.assert_allclose(soundd.get_webrtc_audio(2), [0., 0.5])
    np.testing.assert_allclose(soundd.get_webrtc_audio(2), [-0.5, 32767 / 32768])
    np.testing.assert_allclose(soundd.get_webrtc_audio(2), [0., 0.])

  def test_webrtc_audio_rejects_wrong_sample_rate(self, mocker):
    mocker.patch.object(Soundd, "load_sounds")
    soundd = Soundd()
    soundd.add_webrtc_audio(bytes(4), SAMPLE_RATE // 2)
    assert not soundd.webrtc_audio

  def test_webrtc_audio_drops_stale_samples(self, mocker):
    mocker.patch.object(Soundd, "load_sounds")
    soundd = Soundd()
    samples = np.zeros(WEBRTC_MAX_BUFFER + 100, dtype=np.int16)

    soundd.add_webrtc_audio(samples.tobytes(), SAMPLE_RATE)

    assert len(soundd.webrtc_audio) == WEBRTC_MAX_BUFFER

  def test_publishes_actual_speaker_reference(self, mocker):
    mocker.patch.object(Soundd, "load_sounds")
    soundd = Soundd()
    output = np.empty((2, 1), dtype=np.float32)
    soundd.get_sound_data = mocker.Mock(return_value=np.array([0.25, -0.25], dtype=np.float32))
    soundd.get_webrtc_audio = mocker.Mock(return_value=np.array([0.25, -0.25], dtype=np.float32))
    soundd.callback(output, 2, None, None)
    pm = mocker.Mock()

    soundd.publish_speaker_reference(pm)

    pm.send.assert_called_once()
    service, msg = pm.send.call_args.args
    assert service == "webRtcAudioReference"
    np.testing.assert_array_equal(np.frombuffer(msg.webRtcAudioReference.data, dtype=np.int16), [16383, -16383])

  def test_speaker_volume(self, mocker):
    mocker.patch.object(Soundd, "load_sounds")
    soundd = Soundd()
    volume_sock = mocker.Mock()
    test_cases = ((100, [0.5, -0.5]), (0, [0., 0.]), (50, [0.25, -0.25]))
    volume_messages = []
    for percent, _ in test_cases:
      volume = messaging.new_message("speakerVolume")
      volume.speakerVolume.volume = percent
      volume_messages.append([volume.as_reader()])
    mocker.patch.object(messaging, "drain_sock", side_effect=volume_messages)
    soundd.webrtc_playing = True

    for percent, expected in test_cases:
      soundd.add_webrtc_audio(np.array([16384, -16384], dtype=np.int16).tobytes(), SAMPLE_RATE)

      soundd.update_speaker_volume(volume_sock)

      assert soundd.speaker_volume == percent / 100.
      np.testing.assert_allclose(soundd.get_webrtc_audio(2), expected)

  def test_check_selfdrive_timeout_alert(self, mocker):
    sm = SubMaster(['selfdriveState'])
    pm = PubMaster(['selfdriveState'])

    cs = messaging.new_message('selfdriveState')
    cs.selfdriveState.enabled = True
    threading.Timer(0.01, pm.send, args=("selfdriveState", cs)).start()
    sm.update(100)
    assert sm.updated['selfdriveState']

    received_at = sm.recv_time['selfdriveState']
    clock = mocker.patch("openpilot.selfdrive.ui.soundd.time.monotonic", return_value=received_at + SELFDRIVE_STATE_TIMEOUT)
    assert not check_selfdrive_timeout_alert(sm)

    clock.return_value = received_at + SELFDRIVE_STATE_TIMEOUT + 0.1
    assert check_selfdrive_timeout_alert(sm)

    clock.return_value = received_at + SELFDRIVE_STATE_TIMEOUT + 10
    assert not check_selfdrive_timeout_alert(sm)

  # TODO: add test with micd for checking that soundd actually outputs sounds
