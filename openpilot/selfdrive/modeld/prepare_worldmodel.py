#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

RUN_ID = '3b53ed52-c1d7-4765-8069-5bd6109d86ce'
CHECKPOINT = 15360


def prepare(directory):
  import torch
  from safetensors.torch import save_file

  directory.mkdir(parents=True, exist_ok=True)
  config_path = directory / 'hparams.json'
  if not config_path.exists():
    with urlopen(f'https://reporterv2.comma.life/api/runs/{RUN_ID}/hparams') as response:
      config_path.write_bytes(response.read())
  config = json.loads(config_path.read_text())
  encoder_id = config['dataloader']['compressor_model']
  encoder_path = directory / 'encoder' / 'encoder.onnx'
  if not encoder_path.exists():
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    registry = 'http://models.comma.internal:8081'
    with urlopen(f'{registry}/metaexperiment/{encoder_id}') as response:
      metadata = json.load(response)
    urlretrieve(f"{registry}/checkpoint/{encoder_id}/{metadata['last_epoch']}/encoder.onnx", encoder_path.with_suffix('.partial'))
    encoder_path.with_suffix('.partial').rename(encoder_path)

  weights_path = directory / 'weights.fp8.safetensors'
  if weights_path.exists():
    return
  package = directory / 'model.bf16.torchpackage'
  if not package.exists():
    url = f'http://data-gen.comma.life:3080/reporterv2/checkpoint/{RUN_ID}/{CHECKPOINT}/{package.name}'
    urlretrieve(url, package.with_suffix('.partial'))
    package.with_suffix('.partial').rename(package)
  with ZipFile(package) as archive:
    archive.extract('archive/assets/state_dict.pt', directory)
  torch.set_num_threads(8)
  state = torch.load(directory / 'archive/assets/state_dict.pt', map_location='cpu', mmap=True, weights_only=True)
  for name, value in list(state.items()):
    if name.startswith('blocks.') and name.endswith('.weight') and value.ndim == 2:
      value = value.float()
      scale = value.abs().amax().clamp_min(1e-12) / 448.0
      state[name] = (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
      state[name.removesuffix('.weight') + '.weight_scale'] = scale
  save_file(state, weights_path.with_suffix('.partial'))
  weights_path.with_suffix('.partial').rename(weights_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Prepare the FP8 4B planner and its image encoder; requires reporter access.')
  parser.add_argument('directory', type=Path)
  args = parser.parse_args()
  prepare(args.directory)
