from opendbc.car import CarSpecs, PlatformConfig, Platforms


class CAR(Platforms):
  MOCK = PlatformConfig(
    [],
    CarSpecs(wheelbase=2.7, steerRatio=13),
    {}
  )
