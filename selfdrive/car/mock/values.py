from typing import Dict, List, Optional, Union
from enum import StrEnum

from openpilot.selfdrive.car.docs_definitions import CarInfo


class CAR(StrEnum):
  MOCK = 'mock'


CAR_INFO: Dict[str, Optional[Union[CarInfo, List[CarInfo]]]] = {
  CAR.MOCK: None,
}
