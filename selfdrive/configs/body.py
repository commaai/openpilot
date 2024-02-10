from typing import Dict, Set
from openpilot.selfdrive.configs.base import CORE_SERVICES
from openpilot.selfdrive.configs.threex import ThreexConfig
from openpilot.selfdrive.manager.process import ManagerProcess
from openpilot.selfdrive.manager.process_config import BRIDGE, WEBJOYSTICKD, WEBRTCD


class BodyConfig(ThreexConfig):
  def get_services(self) -> Set[ManagerProcess]:
    return CORE_SERVICES | {BRIDGE, WEBRTCD, WEBJOYSTICKD}

  def get_env(self) -> Dict[str, str]:
    return {}
