from openpilot.selfdrive.configs.base import Processes
from openpilot.selfdrive.configs.three import ThreeConfig
from openpilot.selfdrive.manager.process_config import QCOMGPSD


class ThreexConfig(ThreeConfig):
  GPS_SERVICES: Processes = {QCOMGPSD}
