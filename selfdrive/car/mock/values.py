from enum import StrEnum
from typing import Dict, List, Optional, Union

from openpilot.selfdrive.car.docs_definitions import CarInfo


class CAR(StrEnum):
  MOCK = 'mock'


CAR_INFO: dict[str, CarInfo | list[CarInfo] | None] = {
  CAR.MOCK: None,
}
