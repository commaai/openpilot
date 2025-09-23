import copy
from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.interfaces import CarStateBase
from opendbc.car.rivian.values import DBC, GEAR_MAP
from opendbc.car.common.conversions import Conversions as CV

GearShifter = structs.CarState.GearShifter


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.last_speed = 30

    self.acm_lka_hba_cmd = None
    self.sccm_wheel_touch = None
    self.vdm_adas_status = None

  def update(self, can_parsers) -> structs.CarState:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]
    cp_adas = can_parsers[Bus.adas]
    ret = structs.CarState()

    # Vehicle speed
    ret.vEgoRaw = cp.vl["ESP_Status"]["ESP_Vehicle_Speed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = abs(ret.vEgoRaw) < 0.01
    conversion = CV.KPH_TO_MS if cp_adas.vl["Cluster"]["Cluster_Unit"] == 0 else CV.MPH_TO_MS
    ret.vEgoCluster = cp_adas.vl["Cluster"]["Cluster_VehicleSpeed"] * conversion

    # Gas pedal
    ret.gasPressed = cp.vl["VDM_PropStatus"]["VDM_AcceleratorPedalPosition"] > 0

    # Brake pedal
    ret.brake = cp.vl["ESPiB3"]["ESPiB3_pMC1"] / 250.0  # pressure in Bar
    ret.brakePressed = cp.vl["iBESP2"]["iBESP2_BrakePedalApplied"] == 1

    # Steering wheel
    ret.steeringAngleDeg = cp.vl["EPAS_AdasStatus"]["EPAS_InternalSas"]
    ret.steeringRateDeg = cp.vl["EPAS_AdasStatus"]["EPAS_SteeringAngleSpeed"]
    ret.steeringTorque = cp.vl["EPAS_SystemStatus"]["EPAS_TorsionBarTorque"]
    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > 1.0, 5)

    ret.steerFaultTemporary = cp.vl["EPAS_AdasStatus"]["EPAS_EacErrorCode"] != 0

    # Cruise state
    speed = min(int(cp_adas.vl["ACM_tsrCmd"]["ACM_tsrSpdDisClsMain"]), 85)
    self.last_speed = speed if speed != 0 else self.last_speed
    ret.cruiseState.enabled = cp_cam.vl["ACM_Status"]["ACM_FeatureStatus"] == 1
    # TODO: find cruise set speed on CAN
    ret.cruiseState.speed = self.last_speed * CV.MPH_TO_MS  # detected speed limit
    if not self.CP.openpilotLongitudinalControl:
      ret.cruiseState.speed = -1
    ret.cruiseState.available = True  # cp.vl["VDM_AdasSts"]["VDM_AdasInterfaceStatus"] == 1
    ret.cruiseState.standstill = cp.vl["VDM_AdasSts"]["VDM_AdasVehicleHoldStatus"] == 1

    # TODO: log ACM_Unkown2=3 as a fault. need to filter it at the start and end of routes though
    # ACM_FaultStatus hasn't been seen yet
    ret.accFaulted = (cp_cam.vl["ACM_Status"]["ACM_FaultStatus"] == 1 or
                      # VDM_AdasFaultStatus=Brk_Intv is the default for some reason
                      # VDM_AdasFaultStatus=Imps_Cmd was seen when sending it rapidly changing ACC enable commands
                      # VDM_AdasFaultStatus=Cntr_Fault isn't fully understood, but we've seen it in the wild
                      cp.vl["VDM_AdasSts"]["VDM_AdasFaultStatus"] in (3,))  # 3=Imps_Cmd

    # Gear
    ret.gearShifter = GEAR_MAP.get(int(cp.vl["VDM_PropStatus"]["VDM_Prndl_Status"]), GearShifter.unknown)

    # Doors
    ret.doorOpen = any(cp_adas.vl["IndicatorLights"][door] != 2 for door in ("RearDriverDoor", "FrontPassengerDoor", "DriverDoor", "RearPassengerDoor"))

    # Blinkers
    ret.leftBlinker = cp_adas.vl["IndicatorLights"]["TurnLightLeft"] in (1, 2)
    ret.rightBlinker = cp_adas.vl["IndicatorLights"]["TurnLightRight"] in (1, 2)

    # Seatbelt
    ret.seatbeltUnlatched = cp.vl["RCM_Status"]["RCM_Status_IND_WARN_BELT_DRIVER"] != 0

    # Blindspot
    # ret.leftBlindspot = False
    # ret.rightBlindspot = False

    # AEB
    ret.stockAeb = cp_cam.vl["ACM_AebRequest"]["ACM_EnableRequest"] != 0

    # Messages needed by carcontroller
    self.acm_lka_hba_cmd = copy.copy(cp_cam.vl["ACM_lkaHbaCmd"])
    self.sccm_wheel_touch = copy.copy(cp.vl["SCCM_WheelTouch"])
    self.vdm_adas_status = copy.copy(cp.vl["VDM_AdasSts"])

    return ret

  @staticmethod
  def get_can_parsers(CP):
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 0),
      Bus.adas: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 1),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 2),
    }
