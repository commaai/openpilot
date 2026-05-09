import json
import os
from dataclasses import dataclass
from pathlib import Path

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MODELS_DIR = Path(__file__).resolve().parent / 'models'
COMPILED_FLAGS_PATH = MODELS_DIR / 'tg_compiled_flags.json'

# usbgpu
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001

def set_tinygrad_env() -> None:
  with open(COMPILED_FLAGS_PATH) as f:
    for k, v in json.load(f).items():
      os.environ[k] = str(v)


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
