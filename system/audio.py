import os

from openpilot.common.swaglog import cloudlog


def _get_audio_device_patterns(direction: str) -> list[str]:
  env_var = f"OPENPILOT_AUDIO_{direction.upper()}_DEVICE"
  override = os.getenv(env_var, "").strip()
  if override:
    return [pattern.strip() for pattern in override.split(",") if pattern.strip()]

  return [
    "sdm845-comma-simple-snd-card",
    "comma-simple",
    "sdm845-tavil-snd-card",
  ]


def get_sounddevice_device(sd, *, is_input: bool) -> int | None:
  direction = "input" if is_input else "output"
  channel_key = "max_input_channels" if is_input else "max_output_channels"
  patterns = _get_audio_device_patterns(direction)

  devices = sd.query_devices()
  for pattern in patterns:
    pattern_lower = pattern.lower()
    for idx, dev in enumerate(devices):
      if dev[channel_key] <= 0:
        continue
      if pattern_lower in dev["name"].lower():
        cloudlog.info(f"using {direction} audio device {idx}: {dev['name']}")
        return idx

  cloudlog.warning(f"no preferred {direction} audio device found, falling back to PortAudio default")
  return None
