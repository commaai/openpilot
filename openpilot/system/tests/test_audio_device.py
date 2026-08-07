from openpilot.system.audio_device import AutomaticAudioDevice, device_card_index


def test_device_card_index():
  assert device_card_index({"name": "USB Headset: Audio (hw:3,0)"}) == 3
  assert device_card_index({"name": "pipewire"}) is None


def test_prefers_highest_usb_card_then_new_attachment(mocker):
  sd = mocker.Mock()
  sd.query_devices.return_value = [
    {"name": "onboard (hw:0,0)", "max_output_channels": 2},
    {"name": "older USB (hw:1,0)", "max_output_channels": 2},
    {"name": "newer USB (hw:3,0)", "max_output_channels": 2},
  ]
  cards = mocker.patch("openpilot.system.audio_device.usb_card_indices", return_value={1, 3})
  selector = AutomaticAudioDevice("output")

  assert selector.select(sd) == (2, True)

  sd.query_devices.return_value.append({"name": "just attached (hw:2,0)", "max_output_channels": 2})
  cards.return_value = {1, 2, 3}
  assert selector.select(sd) == (3, True)


def test_mic_only_usb_does_not_replace_output(mocker):
  sd = mocker.Mock()
  sd.query_devices.return_value = [
    {"name": "onboard (hw:0,0)", "max_input_channels": 1, "max_output_channels": 2},
    {"name": "USB mic (hw:2,0)", "max_input_channels": 1, "max_output_channels": 0},
  ]
  mocker.patch("openpilot.system.audio_device.usb_card_indices", return_value={2})

  assert AutomaticAudioDevice("input").select(sd)[0] == 1
  assert AutomaticAudioDevice("output").select(sd)[0] is None
