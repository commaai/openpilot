from typing import Dict, List, Union

from selfdrive.car.docs_definitions import CarInfo


class CAR:
  MOCK = 'mock'


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {}
