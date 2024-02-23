from openpilot.selfdrive.car.body.values import CAR as BODY
from openpilot.selfdrive.car.chrysler.values import CAR as CHRYSLER
from openpilot.selfdrive.car.ford.values import CAR as FORD
from openpilot.selfdrive.car.gm.values import CAR as GM
from openpilot.selfdrive.car.honda.values import CAR as HONDA
from openpilot.selfdrive.car.hyundai.values import CAR as HYUNDAI
from openpilot.selfdrive.car.nissan.values import CAR as NISSAN
from openpilot.selfdrive.car.subaru.values import CAR as SUBARU
from openpilot.selfdrive.car.tesla.values import CAR as TESLA
from openpilot.selfdrive.car.toyota.values import CAR as TOYOTA
from openpilot.selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN


PLATFORMS = BODY | CHRYSLER | FORD | GM | HONDA | HYUNDAI | NISSAN | SUBARU | TESLA | TOYOTA | VOLKSWAGEN

BRANDS = [BODY, CHRYSLER, FORD, GM, HONDA, HYUNDAI, NISSAN, SUBARU, TESLA, TOYOTA, VOLKSWAGEN]

def get_platform(candidate):
  for platform in [platform for brand in BRANDS for platform in brand]:
    if str(platform) == candidate:
      return platform
  return None
