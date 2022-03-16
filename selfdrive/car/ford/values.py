from typing import Dict, List, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo
Ecu = car.CarParams.Ecu

MAX_ANGLE = 87.  # make sure we never command the extremes (0xfff) which cause latching fault


class CAR:
  FUSION = "FORD FUSION 2018"


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.FUSION: CarInfo("Ford Fusion 2018", "All")
}

DBC = {
  CAR.FUSION: dbc_dict('ford_fusion_2018_pt', 'ford_fusion_2018_adas'),
}
