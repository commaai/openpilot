from enum import StrEnum

from openpilot.selfdrive.car.docs_definitions import CarInfo


class CAR(StrEnum):
  MOCK = 'mock'


CAR_INFO: dict[str, CarInfo | list[CarInfo] | None] = {
  CAR.MOCK: None,
}
