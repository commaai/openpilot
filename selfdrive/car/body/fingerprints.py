from cereal import car
from selfdrive.car.body.values import CAR
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu

FINGERPRINTS = {
  CAR.BODY: [{
    513: 8, 516: 8, 514: 3, 515: 4,
  }],
}

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
  ],
)

FW_VERSIONS = {
  CAR.BODY: {
    (Ecu.engine, 0x720, None): [
      b'0.0.01',
      b'02/27/2022',
      b'0.3.00a',
    ],
    # git hash of the firmware used
    (Ecu.debug, 0x721, None): [
      b'166bd860',
      b'dc780f85',
    ],
  },
}
