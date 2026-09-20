import threading

from openpilot.selfdrive.ui.usb_speaker import USBSpeaker, find_usb_output


def test_usb_hotplug_discovery(tmp_path):
  assert find_usb_output(tmp_path) is None
  card = tmp_path / 'card2'
  card.mkdir()
  stream = card / 'stream0'
  stream.write_text('USB Audio\nCapture:\n')
  assert find_usb_output(tmp_path) is None
  stream.write_text('USB Audio\nPlayback:\n')
  assert find_usb_output(tmp_path) == 'plughw:2,0'
  stream.unlink()
  assert find_usb_output(tmp_path) is None
  other = tmp_path / 'card3'
  other.mkdir()
  (other / 'stream0').write_text('USB Audio\nPlayback:\n')
  assert find_usb_output(tmp_path) == 'plughw:3,0'


def test_alsa_stream_lifecycle():
  called = threading.Event()
  def callback(output, frames, time, status):
    output.fill(0)
    called.set()
  speaker = USBSpeaker('null', callback)
  try:
    speaker.start()
    assert called.wait(2)
    assert speaker.active
  finally:
    speaker.close()
    speaker.close()
  assert not speaker.active
  assert not speaker.pcm


def test_usb_disconnect_releases_device(mocker):
  import errno
  speaker = USBSpeaker('null', lambda output, *_: output.fill(0))
  mocker.patch.object(speaker.lib, 'snd_pcm_avail_update', return_value=-errno.ENODEV)
  mocker.patch.object(speaker.lib, 'snd_pcm_recover', return_value=-errno.ENODEV)
  speaker.start()
  speaker.thread.join(timeout=2)
  assert not speaker.thread.is_alive()
  assert not speaker.active
  assert not speaker.pcm
  speaker.close()
