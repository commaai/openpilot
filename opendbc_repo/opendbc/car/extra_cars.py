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
      CommunityCarDocs("Acura ADX 2025-26"),
      CommunityCarDocs("Acura Integra 2023-25"),
      CommunityCarDocs("Acura MDX 2015-16", "Advance Package"),
      CommunityCarDocs("Acura MDX 2017-20"),
      CommunityCarDocs("Acura MDX Hybrid 2017-20"),
      CommunityCarDocs("Acura MDX 2022-24"),
      CommunityCarDocs("Acura RDX 2022-25"),
      CommunityCarDocs("Acura RLX 2017", "Advance Package or Technology Package"),
      CommunityCarDocs("Acura TLX 2015-17", "Advance Package"),
      CommunityCarDocs("Acura TLX 2018-20"),
      CommunityCarDocs("Acura TLX 2022-23"),
      GMSecurityCarDocs("Acura ZDX 2024"),
      CommunityCarDocs("Honda Accord 2016-17", "Honda Sensing"),
      CommunityCarDocs("Honda Accord Hybrid 2017"),
      CommunityCarDocs("Honda Clarity 2018-21"),
      GMSecurityCarDocs("Honda Prologue 2024-25"),
    ],
  )

  EXTRA_HYUNDAI = ExtraPlatformConfig(
    [
      CommunityCarDocs("Hyundai Palisade 2023-24", "Highway Driving Assist II"),
      CommunityCarDocs("Kia Telluride 2023-24", "Highway Driving Assist II"),
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
      FlexRayCarDocs("Audi A4 2016-24"),
      FlexRayCarDocs("Audi A5 2016-24"),
      FlexRayCarDocs("Audi Q5 2017-24"),
    ],
  )
