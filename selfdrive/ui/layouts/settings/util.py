"""
bool hasLongitudinalControl(const cereal::CarParams::Reader &car_params) {
  // Using the experimental longitudinal toggle, returns whether longitudinal control
  // will be active without needing a restart of openpilot
  return car_params.getAlphaLongitudinalAvailable()
             ? Params().getBool("AlphaLongitudinalEnabled")
             : car_params.getOpenpilotLongitudinalControl();
}
"""

def has_longitudinal
