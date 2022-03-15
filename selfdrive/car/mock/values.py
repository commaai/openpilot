from typing import Dict, List, Union

from selfdrive.car import CarInfo


class CAR:
  MOCK = 'mock'


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {}
