import os


def camera120_enabled() -> bool:
  return os.environ.get("CAMERA_720P120") == "1"


def pinball_camera120(CP) -> bool:
  return camera120_enabled() and CP.notCar and CP.brand == "pinball"


# These producers require full-resolution/multiple cameras or driving models.
CAMERA120_DISABLED_PROCESSES = frozenset({
  "encoderd", "loggerd", "webcamerad", "modeld", "dmonitoringmodeld", "dmonitoringd",
  "locationd", "calibrationd", "paramsd", "torqued", "lagd", "plannerd", "radard",
  "maneuversd", "lateral_maneuversd",
})

CAMERA120_UNUSED_SERVICES = (
  "wideRoadCameraState", "cabinCameraState", "modelV2", "driverMonitoringState",
  "extrinsicsCalibration", "longitudinalPlan", "deviceMotion", "lateralDelay",
  "vehicleParameters", "radarState", "lateralTorqueParameters", "driverAssistance",
)
