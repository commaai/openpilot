from openpilot.selfdrive.car import CarSpecs, PlatformConfig, Platforms


class CAR(Platforms):
  MOCK = PlatformConfig(
    'mock',
    [],
    CarSpecs(mass=1700, wheelbase=2.7, steerRatio=13),
    {}
  )
