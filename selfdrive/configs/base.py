import abc
from typing import Dict, Set

from openpilot.selfdrive.manager.process import ManagerProcess

from openpilot.selfdrive.manager.process_config import ATHENA, CALIBRATIOND, LOGCAT, PROCLOG, MODELD, NAVMODELD, LOCATIOND, TORQUED, CONTROLSD, \
                                                       DELETER, NAVD, PARAMSD, PLANNERD, RADARD, TOMBSTONED, UPDATED, UPLOADERD, THERMALD, \
                                                       STATSD, LOGGERD, LOGMESSAGED, ENCODERD, DMONITORINGMODELD, DMONITORINGD, UI, SOUNDD, MICD


Processes = Set[ManagerProcess]
Environment = Dict[str, str]

# services required for driving
CORE_SERVICES: Processes = {
  MODELD,
  NAVMODELD,
  MAPSD,

  LOCATIOND,
  CALIBRATIOND,
  TORQUED,
  CONTROLSD,
  PARAMSD,
  PLANNERD,
  RADARD,
  THERMALD,

  UPDATED,
}

LOGGING_SERVICES: Processes = {
  LOGCAT,
  PROCLOG,
  LOGGERD,
  LOGMESSAGED,
  ENCODERD,
  DELETER,
  TOMBSTONED,
  STATSD,
}

# Services for interacting with the comma api and uploading routes
COMMA_SERVICES: Processes = {
  ATHENA,
  UPLOADERD,
}

DMONITORING_SERVICES: Processes = {
  DMONITORINGMODELD,
  DMONITORINGD,
}

UI_SERVICES: Processes = {
  UI,
  SOUNDD,
  MICD,
  NAVD,
}


class BaseConfig(abc.ABC):
  @abc.abstractmethod
  def get_services(self) -> Processes:
    pass

  @abc.abstractmethod
  def get_env(self) -> Environment:
    pass
