import json
import os
from dataclasses import dataclass
from pathlib import Path

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MODELS_DIR = Path(__file__).resolve().parent / 'models'
COMPILED_FLAGS_PATH = MODELS_DIR / 'tg_compiled_flags.json'


def set_tinygrad_backend_from_compiled_flags() -> None:
  if os.path.isfile(COMPILED_FLAGS_PATH):
    with open(COMPILED_FLAGS_PATH) as f:
      os.environ['DEV'] = str(json.load(f)['DEV'])


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
