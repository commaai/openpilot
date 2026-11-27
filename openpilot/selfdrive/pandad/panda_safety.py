from opendbc.car.structs import CarParams, car
from panda import Panda
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

SafetyModel = CarParams.SafetyModel


class PandaSafety:
  """Sets the panda's safety mode onroad: ELM327 for fingerprinting, then the car's safety model once card is ready."""

  def __init__(self, panda: Panda):
    self.panda = panda
    self.params = Params()
    self.initialized = False
    self.log_once = False
    self.safety_configured = False
    self.prev_obd_multiplexing = False
    self.not_car = False  # from the CarParams, for the IR LEDs

  def configure_safety_mode(self, is_onroad: bool) -> None:
    if is_onroad and not self.safety_configured:
      self.update_multiplexing_mode()

      car_params = self.fetch_car_params()
      if car_params:
        cloudlog.warning(f"got {len(car_params)} bytes CarParams")
        self.set_safety_mode(car_params)
        self.safety_configured = True
    elif not is_onroad:
      self.initialized = False
      self.safety_configured = False
      self.log_once = False
      self.not_car = False

  def update_multiplexing_mode(self) -> None:
    # Initialize to ELM327 without OBD multiplexing for initial fingerprinting
    if not self.initialized:
      self.prev_obd_multiplexing = False
      self.panda.set_safety_mode(SafetyModel.elm327, 1)
      self.initialized = True

    # Switch between multiplexing modes based on the OBD multiplexing request
    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      self.panda.set_safety_mode(SafetyModel.elm327, 0 if obd_multiplexing_requested else 1)
      self.prev_obd_multiplexing = obd_multiplexing_requested
      self.params.put_bool("ObdMultiplexingChanged", True, block=True)

  def fetch_car_params(self) -> bytes:
    if not self.params.get_bool("FirmwareQueryDone"):
      return b""

    if not self.log_once:
      cloudlog.warning("Finished FW query, Waiting for params to set safety model")
      self.log_once = True

    if not self.params.get_bool("ControlsReady"):
      return b""
    return self.params.get("CarParams") or b""

  def set_safety_mode(self, params_bytes: bytes) -> None:
    safety_model, safety_param, alternative_experience = SafetyModel.noOutput, 0, 0
    try:
      with car.CarParams.from_bytes(params_bytes) as CP:
        self.not_car = CP.notCar
        # the panda takes the 16 bit value as is
        alternative_experience = CP.alternativeExperience & 0xFFFF
        safety_model, safety_param = CP.safetyConfigs[0].safetyModel.raw, CP.safetyConfigs[0].safetyParam
    except Exception:
      # unparsable or no safety config: noOutput, set once instead of failing every cycle
      cloudlog.exception("pandad.invalid_car_params")

    cloudlog.warning(f"setting safety model: {safety_model}, param: {safety_param}, alternative experience: {alternative_experience}")
    # the panda only takes the alternative experience while it's not in a car safety mode, so it goes first
    self.panda.set_alternative_experience(alternative_experience)
    self.panda.set_safety_mode(safety_model, safety_param)
