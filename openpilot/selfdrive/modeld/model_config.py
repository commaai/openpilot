import json
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

CAMERA_CONFIGS = [(c.width, c.height) for c in (_ar_ox_fisheye, _os_fisheye)]


def frame_config(width, height):
  return [width, height, *get_nv12_info(width, height)]


def driving_model_args(frame_skip):
  args = ['--float32', '--out-of-band']
  for width, height in CAMERA_CONFIGS:
    inputs: dict = {name: {
      'warp': {'frame': frame_config(width, height), 'output_size': MEDMODEL_INPUT_SIZE, 'layout': 'yuv420', 'transform': transform},
      'history': {'axis': 1, 'size': 6, 'stride': frame_skip},
    } for name, transform in [('img', 'tfm'), ('big_img', 'big_tfm')]}
    inputs.update({
      'features_buffer': {'source': 'prev_feat', 'history': {'axis': 1, 'stride': frame_skip, 'delay': frame_skip-1}},
      'desire_pulse': {'source': 'desire', 'history': {'axis': 1, 'stride': frame_skip, 'reduce': 'max'}},
    })
    config = {'inputs': inputs, 'pack': ['tfm', 'big_tfm', 'desire', 'traffic_convention', 'action_t', 'prev_feat', 'img', 'big_img']}
    args += ['--config', f'{width}x{height}={json.dumps(config, separators=(",", ":"))}']
  return args


def warp_args(width, height, output_size, layout, border_fill=None):
  args = ['--frame', ','.join(map(str, frame_config(width, height))),
          '--warp-to', f'{output_size[0]}x{output_size[1]}', '--layout', layout]
  if border_fill is not None:
    args += ['--border-fill', str(border_fill)]
  return args
