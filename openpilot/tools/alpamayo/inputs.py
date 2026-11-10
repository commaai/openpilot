import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeModel, Cosmos3EdgeVisionEmbeddings
from alpamayo2_edge.chat_template import build_conversation
from alpamayo2_edge.config import Alpamayo2EdgeConfig, build_alpamayo2_edge_tokenizer
from alpamayo2_edge.helper import _tokenize_messages, get_processor, prepare_model_inputs
from alpamayo2_edge.models.delta_tokenizer import DeltaTrajectoryTokenizer
from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS, CAMERA_PIXEL_BOUNDS


def synthetic_sample(profile):
  cameras = (1, 6) if profile == 'openpilot' else tuple(CAMERA_FRAME_COUNTS)
  indices, timestamps = [], []
  for camera in cameras:
    offsets = (-2., -1., -.5, -.1, 0.) if camera in (1, 6) else (-1., -.1, 0.)
    indices.extend([camera] * len(offsets))
    timestamps.extend(int((10 + offset) * 1e6) for offset in offsets)
  xyz = torch.zeros(1, 20, 3)
  xyz[0, :, 0] = torch.arange(-19, 1) * 1.5
  return {
    'image_frames': torch.full((len(indices), 1, 3, 288, 576), 127, dtype=torch.uint8),
    'camera_indices': torch.tensor(indices), 'absolute_timestamps': torch.tensor(timestamps),
    'ego_history_xyz': xyz, 'ego_history_rot': torch.eye(3).expand(1, 20, 3, 3).contiguous(),
    'nav_text': ['Continue straight'],
  }


def load_sample(path):
  with np.load(path, allow_pickle=False) as data:
    return {key: [str(data[key].item())] if key == 'nav_text' else torch.from_numpy(data[key].copy()) for key in data.files}


def prepare(checkpoint, data, profile):
  config = Alpamayo2EdgeConfig(**json.loads((Path(checkpoint) / 'config.json').read_text()))
  config._name_or_path = str(Path(checkpoint).resolve())
  tokenizer = build_alpamayo2_edge_tokenizer(config._name_or_path)
  if profile == 'nvidia':
    prepared = prepare_model_inputs(data, config, tokenizer)
  else:
    assert data['camera_indices'].tolist() == [1]*5 + [6]*5
    assert data['image_frames'].dtype == torch.uint8 and data['image_frames'].shape[:3] == (10, 1, 3)
    assert data['ego_history_xyz'].shape == (1, 20, 3) and data['ego_history_rot'].shape == (1, 20, 3, 3)
    # Front wide and narrow only; the absent surround views are never fabricated.
    canonical = dict(data, camera_pixel_bounds=CAMERA_PIXEL_BOUNDS)
    tokenized = _tokenize_messages(build_conversation(canonical), get_processor(tokenizer, config), include_closed_thinking=True)
    prepared = {'tokenized_data': tokenized, 'ego_history_xyz': data['ego_history_xyz'].unsqueeze(0),
                'ego_history_rot': data['ego_history_rot'].unsqueeze(0)}
  return config, tokenizer, prepared


def native_inputs(checkpoint, config, prepared):
  tokenized = prepared['tokenized_data']
  ids = tokenized['input_ids'].clone()
  xyz = prepared['ego_history_xyz'][:, 0]
  rot = prepared['ego_history_rot'][:, 0]
  history = DeltaTrajectoryTokenizer(pad_origin_at_beginning=False).encode(None, None, xyz, rot)
  ids[ids == config.traj_ids['history_pad']] = (history + config.traj_ids['history_id0']).flatten()
  rope_model = SimpleNamespace(config=config.vlm_config)
  rope_model.get_vision_position_ids = partial(Cosmos3EdgeModel.get_vision_position_ids, rope_model)
  positions, delta = Cosmos3EdgeModel.get_rope_index(rope_model, ids,
                                                     tokenized['mm_token_type_ids'], image_grid_thw=tokenized['image_grid_thw'])
  name = 'vlm.model.visual.embeddings.position_embedding.weight'
  index = json.loads((Path(checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']
  with safe_open(Path(checkpoint) / index[name], framework='pt') as f:
    weight = f.get_tensor(name).reshape(16, 16, -1)
  vision_positions = Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings(weight, tokenized['image_grid_thw'], 2)
  return ids.numpy(), positions.numpy(), int(delta.item()), vision_positions.float().numpy()


def rotary(positions):
  frequencies = 1 / (1e8 ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
  angles = positions[..., None].astype(np.float32) * frequencies
  combined = angles[0].copy()
  for axis in (1, 2):
    combined[..., axis:60:3] = angles[axis, ..., axis:60:3]
  combined = np.concatenate((combined, combined), axis=-1)[:, None]
  return np.cos(combined), np.sin(combined)
