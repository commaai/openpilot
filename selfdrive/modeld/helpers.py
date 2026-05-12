import json
from dataclasses import dataclass
from pathlib import Path

from openpilot.selfdrive.modeld.get_model_metadata import metadata_path_for
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'

USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001


def get_tg_input_devices(process_name: str, usbgpu: bool) -> dict[str, str]:
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['usbgpu' if usbgpu else 'default']


def usbgpu_present() -> bool:
  for d in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if int((d / "idVendor").read_text(), 16) == USBGPU_VID and \
         int((d / "idProduct").read_text(), 16) == USBGPU_PID:
        return True
    except (FileNotFoundError, NotADirectoryError, ValueError):
      pass
  return False


@dataclass
class ModeldCompileConfig:
  cam_w: int
  cam_h: int
  prepare_only: bool
  usbgpu: bool = False

  @property
  def big_prefix(self):
    return 'big_' if self.usbgpu else ''

  @property
  def driving_policy_onnx(self):
    return MODELS_DIR / f'{self.big_prefix}driving_policy.onnx'

  @property
  def driving_vision_onnx(self):
    return MODELS_DIR / f'{self.big_prefix}driving_policy.onnx'

  @property
  def vision_metadata(self):
    return metadata_path_for(self.driving_vision_onnx)

  @property
  def policy_metadata(self):
    return metadata_path_for(self.driving_policy_onnx)

  @property
  def nv12(self):
    return (self.cam_w, self.cam_h, *get_nv12_info(self.cam_w, self.cam_h))

  @property
  def pkl_path(self):
    # TODO why str?
    return str(MODELS_DIR / f'{self.big_prefix}{"warp_" if self.prepare_only else ""}{self.cam_w}x{self.cam_h}_tinygrad.pkl')


@dataclass
class WarpCompileConfig:
  cam_w: int
  cam_h: int

  @property
  def nv12(self):
    return (self.cam_w, self.cam_h, *get_nv12_info(self.cam_w, self.cam_h))

  @property
  def pkl_path(self):
    return MODELS_DIR / f'dm_warp_{self.cam_w}x{self.cam_h}_tinygrad.pkl'
