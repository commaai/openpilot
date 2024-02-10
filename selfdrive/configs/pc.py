import os
from openpilot.selfdrive.manager.process import PythonProcess
from openpilot.selfdrive.configs.base import CORE_SERVICES, DMONITORING_SERVICES, UI_SERVICES, BaseConfig, Processes, Environment
from openpilot.selfdrive.manager.process_config import BOARDD, MAPSD, PANDAD, always_run, driverview


class PCConfig(BaseConfig):
  def get_services(self) -> Processes:
    services = CORE_SERVICES | UI_SERVICES | DMONITORING_SERVICES
    if "MAPBOX_TOKEN" not in os.environ:
      services -= {MAPSD}

    return services

  def get_env(self) -> Environment:
    return {}


METADRIVE_BRIDGE = PythonProcess("bridge", "tools.sim.run_bridge", always_run)
METADRIVE_SERVICES: Processes = {METADRIVE_BRIDGE}

class MetaDriveConfig(PCConfig):
  def __init__(self):
    super().__init__()

  def get_services(self) -> Processes:
    return (super().get_services() | METADRIVE_SERVICES) - DMONITORING_SERVICES

  def get_env(self) -> Environment:
    return {
      "PASSIVE": "0",
      "NOBOARD": "1",
      "SIMULATION": "1",
      "SKIP_FW_QUERY": "1",
      "FINGERPRINT": "HONDA CIVIC 2016"
    }


WEBCAM_CAMERAD = PythonProcess("camerad", "tools.webcam.camerad", driverview)

class PCWebcamConfig(PCConfig):
  def get_services(self) -> Processes:
    return (super().get_services() | {WEBCAM_CAMERAD, BOARDD, PANDAD})

  def get_env(self) -> Environment:
    return {}
