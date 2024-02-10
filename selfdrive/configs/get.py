import os
from typing import Dict, Type
from openpilot.selfdrive.configs.base import BaseConfig
from openpilot.selfdrive.configs.pc import MetaDriveConfig, PCWebcamConfig
from openpilot.selfdrive.configs.three import ThreeConfig
from openpilot.selfdrive.configs.threex import ThreexConfig
from openpilot.system.hardware import PC, SIM, TICI, TIZI


CONFIG_OVERRIDE_MAP: Dict[str, Type[BaseConfig]] = {
  "sim": MetaDriveConfig,
  "webcam": PCWebcamConfig,
  "three": ThreeConfig,
  "threex": ThreexConfig
}


def get_config():
  CONFIG_OVERRIDE = os.environ.get("CONFIG", None)

  if CONFIG_OVERRIDE is not None:
    config = CONFIG_OVERRIDE_MAP.get(CONFIG_OVERRIDE, None)
    if not config:
      raise ValueError(f"Unknown CONFIG_OVERRIDE: {CONFIG_OVERRIDE}")
    return config
  else:
    if PC and SIM:
      return MetaDriveConfig()
    elif PC:
      return PCWebcamConfig()
    elif TIZI:
      return ThreexConfig()
    elif TICI:
      return ThreeConfig()
    else:
      raise ValueError("No config available for system hardware")

CONFIG: BaseConfig = get_config()
