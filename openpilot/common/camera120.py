import os


# Keep the legacy module/flag compatible with existing device startup and clients.
CAMERA_FPS = 60


def camera120_enabled() -> bool:
  return os.environ.get("CAMERA_720P60", os.environ.get("CAMERA_720P120")) == "1"


CAMERA120_UNUSED_SERVICES = (
  "wideRoadCameraState", "cabinCameraState", "modelV2", "driverMonitoringState",
  "extrinsicsCalibration", "longitudinalPlan", "deviceMotion", "lateralDelay",
  "vehicleParameters", "radarState", "lateralTorqueParameters", "driverAssistance",
)
