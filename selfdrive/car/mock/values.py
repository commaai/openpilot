from typing import Dict, List, Optional, Union

from selfdrive.car.docs_definitions import CarInfo


class CAR:
  MOCK = 'mock'


CAR_INFO: Dict[str, Optional[Union[CarInfo, List[CarInfo]]]] = {
  CAR.MOCK: None,
}
