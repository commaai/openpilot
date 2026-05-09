import json
from dataclasses import dataclass
from pathlib import Path

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'

# usbgpu
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
class CompileConfig:
  cam_w: int
  cam_h: int
  prepare_only: bool
  prefix: str

  @property
  def pkl_path(self):
    return str(MODELS_DIR / f'{self.prefix}{"warp_" if self.prepare_only else ""}{self.cam_w}x{self.cam_h}_tinygrad.pkl')

  @property
  def nv12(self):
    return (self.cam_w, self.cam_h, *get_nv12_info(self.cam_w, self.cam_h))
