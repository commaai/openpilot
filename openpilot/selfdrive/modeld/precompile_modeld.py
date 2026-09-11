#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from openpilot.common.basedir import BASEDIR
from openpilot.common.file_chunker import chunk_file, get_chunk_targets
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import MODELS_DIR


def compiler_hashes():
  root = Path(BASEDIR)
  sources = [
    'openpilot/selfdrive/modeld/compile_modeld.py',
    'openpilot/selfdrive/modeld/precompile_modeld.py',
    'openpilot/selfdrive/modeld/get_model_metadata.py',
    'openpilot/selfdrive/modeld/helpers.py',
    'openpilot/selfdrive/modeld/constants.py',
    'openpilot/common/transformations/camera.py',
    'openpilot/common/transformations/model.py',
    'openpilot/common/hardware/hw.py',
    'openpilot/system/camerad/cameras/nv12_info.py',
  ]
  hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
  digest = hashlib.sha256()
  for path in sorted((root / 'tinygrad_repo/tinygrad').rglob('*.py')):
    digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
  hashes['tinygrad_repo'] = digest.hexdigest()
  return hashes


if __name__ == '__main__':
  # ONNX metadata parsing must not claim the USB GPU in the parent process.
  os.environ['DEV'] = 'CPU'
  parser = argparse.ArgumentParser(description='Precompile and package the big model for USB AMD.')
  parser.add_argument('onnx', type=Path, help='ONNX exported by xx/ml_tools/openpilot_compile/compile_torchtitan_supercombo.py')
  args = parser.parse_args()
  source_hashes = compiler_hashes()
  cameras = [(c.width, c.height) for c in (_ar_ox_fisheye, _os_fisheye)]
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  env = os.environ | {
    'DEV': 'USB+AMD:LLVM', 'FRAME_DEV': 'CPU', 'FLOAT16': '1', 'JIT_BATCH_SIZE': '0',
    'GMMU': '0', 'TC_OPT': '2', 'TC_OCCUPANCY_OPT': '1',
    'HCQDEV_WAIT_TIMEOUT_MS': os.getenv('HCQDEV_WAIT_TIMEOUT_MS', '300000'),
  }
  with tempfile.TemporaryDirectory() as directory:
    compiled = Path(directory) / 'big_driving_tinygrad.pkl'
    subprocess.run([
      sys.executable, str(Path(__file__).with_name('compile_modeld.py')),
      '--model-size', 'x'.join(map(str, MEDMODEL_INPUT_SIZE)),
      '--camera-resolutions', *(f'{w}x{h}' for w, h in cameras),
      '--onnx', str(args.onnx.resolve()), '--output', str(compiled),
      '--frame-skip', str(frame_skip), '--benchmark-runs', '20',
    ], env=env, check=True)
    assert compiler_hashes() == source_hashes, 'Compiler sources changed during precompilation'
    with args.onnx.open('rb') as f:
      onnx_sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
    with compiled.open('rb') as f:
      pickle_sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
    targets = get_chunk_targets(compiled, compiled.stat().st_size)
    chunk_file(compiled, targets)
    for target in targets:
      shutil.copyfile(target, MODELS_DIR / Path(target).name)
    names = {Path(target).name for target in targets}
    for old_chunk in MODELS_DIR.glob('big_driving_tinygrad.pkl.chunk[0-9]*'):
      if old_chunk.name not in names:
        old_chunk.unlink()
    from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
    metadata = {
      'model_checkpoint': make_metadata_dict(str(args.onnx))['model_checkpoint'],
      'onnx_sha256': onnx_sha256, 'pickle_sha256': pickle_sha256,
      'compiler_sha256': source_hashes, 'camera_resolutions': cameras, 'frame_skip': frame_skip,
      'chunks': {Path(target).name: Path(target).stat().st_size for target in targets[1:]},
    }
    (MODELS_DIR / 'big_driving_tinygrad.json').write_text(json.dumps(metadata, indent=2) + '\n')
