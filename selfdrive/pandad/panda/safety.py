from panda import Panda
import capnp
from cereal import car
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

class PandaSafetyManager:
  def __init__(self, pandas: list[Panda], lock):
    self.pandas = pandas
    self.params = Params()
    self.safety_configured = False
    self.initialized = False
    self.prev_obd_multiplexing = False
    self.log_once = False
    self.lock = lock

  def configure_safety_mode(self):
    is_onroad = self.params.get_bool("IsOnroad")

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

  def update_multiplexing_mode(self):
    # Initialize to ELM327 without OBD multiplexing for initial fingerprinting
    if not self.initialized:
      self.prev_obd_multiplexing = False
      with self.lock:
        for panda in self.pandas:
          panda.set_safety_mode(car.CarParams.SafetyModel.elm327, 1)
      self.initialized = True

    # Switch between multiplexing modes based on OBD multiplexing request
    obd_multiplexing_requested = self.params.get_bool("ObdMultiplexingEnabled")
    if obd_multiplexing_requested != self.prev_obd_multiplexing:
      with self.lock:
        for i, panda in enumerate(self.pandas):
          safety_param = 1 if i > 0 or not obd_multiplexing_requested else 0
          panda.set_safety_mode(car.CarParams.SafetyModel.elm327, safety_param)
      self.prev_obd_multiplexing = obd_multiplexing_requested
      self.params.put_bool("ObdMultiplexingChanged", True)

  def fetch_car_params(self) -> bytes:
    if not self.params.get_bool("FirmwareQueryDone"):
      return b""

    if not self.log_once:
      cloudlog.warning("Finished FW query, waiting for params to set safety model")
      self.log_once = True

    if not self.params.get_bool("ControlsReady"):
      return b""
    return self.params.get("CarParams", encoding='utf-8') or b""

  def set_safety_mode(self, params_bytes: bytes):
    # Parse CarParams from bytes
    try:
      car_params = car.CarParams.from_bytes(params_bytes)
    except capnp.lib.capnp.KjException as e:
      cloudlog.error(f"Failed to parse CarParams: {e}")
      return

    safety_configs = car_params.safetyConfigs
    # alternative_experience = car_params.alternativeExperience

    with self.lock:
      for i, panda in enumerate(self.pandas):
        # Default to SILENT if no config for this panda
        safety_model = car.CarParams.SafetyModel.silent
        safety_param = 0
        if i < len(safety_configs):
          safety_model = safety_configs[i].safetyModel
          safety_param = safety_configs[i].safetyParam

        # panda.set_alternative_experience(alternative_experience)
        panda.set_safety_mode(safety_model, safety_param)
