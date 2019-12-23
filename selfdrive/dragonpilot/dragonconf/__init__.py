#!/usr/bin/env python2.7
from common.params import Params, put_nonblocking

default_conf = {
  'DragonEnableDashcam': '1',
  'DragonEnableDriverSafetyCheck': '1',
  'DragonAutoShutdownAt': '30', # in minute
  'DragonEnableSteeringOnSignal': '0',
  'DragonEnableLogger': '1',
  'DragonEnableUploader': '1',
  'DragonNoctuaMode': '0',
  'DragonCacheCar': '0',
  'DragonCachedModel': '', # for cache car
  'DragonCachedFP': '', # for cache car
  'DragonCachedVIN': '', # for cache car
  'DragonAllowGas': '0',
  'DragonToyotaStockDSU': '0',
  'DragonLatCtrl': '1',
  'DragonUISpeed': '1',
  'DragonUIEvent': '0',
  'DragonUIMaxSpeed': '0',
  'DragonUIFace': '0',
  'DragonUIDev': '0',
  'DragonUIDevMini': '1',
  # 3rd party app
  'DragonEnableTomTom': '0',
  'DragonBootTomTom': '0',
  'DragonRunTomTom': '0',
  'DragonEnableAutonavi': '0',
  'DragonBootAutonavi': '0',
  'DragonRunAutonavi': '0',
  'DragonEnableAegis': '0',
  'DragonBootAegis': '0',
  'DragonRunAegis': '0',
  'DragonEnableMixplorer': '0',
  'DragonRunMixplorer': '0',
  'DragonSteeringMonitorTimer': '3',
  'DragonCameraOffset': '6',
  'DragonUIVolumeBoost': '0',
  'DragonGreyPandaMode': '0',
  'DragonDrivingUI': '1',
  'DragonDisplaySteeringLimitAlert': '1',
  'DragonChargingCtrl': '0',
  'DragonCharging': 70,
  'DragonDisCharging': 60,
  'DragonToyotaLaneDepartureWarning': '1',
  'DragonUILane': '1',
  'DragonUILead': '1',
  'DragonUIPath': '1',
  'DragonUIBlinker': '0',
  'DragonEnableDriverMonitoring': '1',
  'DragonCarModel': '',
  'DragonCarVIN': '',
  'DragonEnableSlowOnCurve': '1',
  'DragonEnableLeadCarMovingAlert': '0',
  'DragonToyotaSnGMod': '0',
}

deprecated_conf = {
  'DragonDisableDriverSafetyCheck': 'DragonEnableDriverSafetyCheck',
  'DragonTempDisableSteerOnSignal': 'DragonEnableSteeringOnSignal',
  'DragonDisableLogger': 'DragonEnableLogger',
  'DragonDisableUploader': 'DragonEnableUploader',
  'DragonBBUI': 'DragonUIDev',
}

# deprecated_conf_invert = {
#   'DragonDisableDriverSafetyCheck': True,
#   'DragonTempDisableSteerOnSignal': False,
#   'DragonDisableLogger': True,
#   'DragonDisableUploader': True,
#   'DragonBBUI': False
# }

def dragonpilot_set_params(params):
  # remove deprecated params
  # for old, new in deprecated_conf.items():
  #   if params.get(old) is not None:
  #     if new is not None:
  #       old_val = str(params.get(old))
  #       new_val = old_val
  #       # invert the value if true
  #       if old in deprecated_conf_invert and deprecated_conf_invert[old] is True:
  #         new_val = "1" if old_val == "0" else "0"
  #       put_nonblocking(new, new_val)
  #     params.delete(old)

  # set params
  for key, val in default_conf.items():
    if params.get(key) is None and key not in deprecated_conf:
      put_nonblocking(key, str(val))

if __name__ == "__main__":
  params = Params()
  params.manager_start()

  dragonpilot_set_params(params)