from openpilot.selfdrive.car import CarSpecs, PlatformConfig, Platforms


class CAR(Platforms):
  MOCK = PlatformConfig(
    car_docs=[],
    specs=CarSpecs(mass=1700, wheelbase=2.7, steerRatio=13),
    dbc_dict={}
  )
