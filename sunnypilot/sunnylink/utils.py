from sunnypilot.sunnylink.api import SunnylinkApi, UNREGISTERED_SUNNYLINK_DONGLE_ID
from openpilot.common.params import Params
from openpilot.system.version import is_prebuilt


def get_sunnylink_status(params=None) -> tuple[bool, bool]:
  """Get the status of Sunnylink on the device. Returns a tuple of (is_sunnylink_enabled, is_registered)."""
  params = params or Params()
  is_sunnylink_enabled = params.get_bool("SunnylinkEnabled")
  is_registered = params.get("SunnylinkDongleId", encoding='utf-8') not in (None, UNREGISTERED_SUNNYLINK_DONGLE_ID)
  return is_sunnylink_enabled, is_registered


def sunnylink_ready(params=None) -> bool:
  """Check if the device is ready to communicate with Sunnylink. That means it is enabled and registered."""
  params = params or Params()
  is_sunnylink_enabled, is_registered = get_sunnylink_status(params)
  return is_sunnylink_enabled and is_registered


def use_sunnylink_uploader(params) -> bool:
  """Check if the device is ready to use Sunnylink and the uploader is enabled."""
  return sunnylink_ready(params) and params.get_bool("EnableSunnylinkUploader")


def sunnylink_need_register(params=None) -> bool:
  """Check if the device needs to be registered with Sunnylink."""
  params = params or Params()
  is_sunnylink_enabled, is_registered = get_sunnylink_status(params)
  return is_sunnylink_enabled and not is_registered


def register_sunnylink():
  """Register the device with Sunnylink if it is enabled."""
  extra_args = {}

  if not Params().get_bool("SunnylinkEnabled"):
    print("Sunnylink is not enabled. Exiting.")
    exit(0)

  if not is_prebuilt():
    extra_args = {
      "verbose": True,
      "timeout": 60
    }

  sunnylink_id = SunnylinkApi(None).register_device(None, **extra_args)
  print(f"SunnyLinkId: {sunnylink_id}")
