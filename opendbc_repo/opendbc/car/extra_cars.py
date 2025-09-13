from dataclasses import dataclass

from opendbc.car import structs, Platforms, ExtraPlatformConfig
from opendbc.car.docs_definitions import ExtraCarDocs, SupportType


@dataclass
class CommunityCarDocs(ExtraCarDocs):
  def init_make(self, CP: structs.CarParams):
    self.support_type = SupportType.COMMUNITY
    self.support_link = "#community"


@dataclass
class ToyotaSecurityCarDocs(ExtraCarDocs):
  def init_make(self, CP: structs.CarParams):
    self.support_type = SupportType.INCOMPATIBLE
    self.support_link = "#can-bus-security"


@dataclass
class GMSecurityCarDocs(ExtraCarDocs):
  def init_make(self, CP: structs.CarParams):
    self.support_type = SupportType.INCOMPATIBLE
    self.support_link = "#can-bus-security"


@dataclass
class FlexRayCarDocs(ExtraCarDocs):
  def init_make(self, CP: structs.CarParams):
    self.support_type = SupportType.INCOMPATIBLE
    self.support_link = "#flexray"


class CAR(Platforms):
  config: ExtraPlatformConfig

  EXTRA_HONDA = ExtraPlatformConfig(
    [
      CommunityCarDocs("Acura Integra 2023-25", "All"),
      CommunityCarDocs("Acura MDX 2015-16", "Advance Package"),
      CommunityCarDocs("Acura MDX 2017-20", "All"),
      CommunityCarDocs("Acura RDX 2022-25", "All"),
      CommunityCarDocs("Acura RLX 2017", "Advance Package or Technology Package"),
      CommunityCarDocs("Acura TLX 2015-17", "Advance Package"),
      CommunityCarDocs("Acura TLX 2018-20", "All"),
      GMSecurityCarDocs("Acura ZDX 2024", "All"),
      CommunityCarDocs("Honda Accord 2016-17", "Honda Sensing"),
      CommunityCarDocs("Honda Clarity 2018-21", "All"),
      CommunityCarDocs("Honda Passport 2026", "All"),
      GMSecurityCarDocs("Honda Prologue 2024-25", "All"),
    ],
  )

  EXTRA_HYUNDAI = ExtraPlatformConfig(
    [
      CommunityCarDocs("Hyundai Palisade 2023-24", package="HDA2"),
      CommunityCarDocs("Kia Telluride 2023-24", package="HDA2"),
    ],
  )

  EXTRA_TOYOTA = ExtraPlatformConfig(
    [
      ToyotaSecurityCarDocs("Subaru Solterra 2023-25"),
      ToyotaSecurityCarDocs("Lexus NS 2022-25"),
      ToyotaSecurityCarDocs("Toyota bZ4x 2023-25"),
      ToyotaSecurityCarDocs("Toyota Camry 2025"),
      ToyotaSecurityCarDocs("Toyota Corolla Cross 2022-25"),
      ToyotaSecurityCarDocs("Toyota Highlander 2025"),
      ToyotaSecurityCarDocs("Toyota RAV4 Prime 2024-25"),
      ToyotaSecurityCarDocs("Toyota Sequoia 2023-25"),
      ToyotaSecurityCarDocs("Toyota Sienna 2024-25"),
      ToyotaSecurityCarDocs("Toyota Tundra 2022-25"),
      ToyotaSecurityCarDocs("Toyota Venza 2021-25"),
    ],
  )

  EXTRA_VOLKSWAGEN = ExtraPlatformConfig(
    [
      FlexRayCarDocs("Audi A4 2016-24", package="All"),
      FlexRayCarDocs("Audi A5 2016-24", package="All"),
      FlexRayCarDocs("Audi Q5 2017-24", package="All"),
    ],
  )
