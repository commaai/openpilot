from dataclasses import dataclass
from pathlib import Path

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

MODELS_DIR = Path(__file__).resolve().parent / 'models'

def get_jit_input_devices(jit) -> dict[str, str]:
  return {name: info[3] for name, info in zip(jit.captured.expected_names, jit.captured.expected_input_info, strict=True)}

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
