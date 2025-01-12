# Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
#
# This file is part of sunnypilot and is licensed under the MIT License.
# See the LICENSE.md file in the root directory for more details.

import os
import pickle
import numpy as np

from cereal import custom
from openpilot.sunnypilot.modeld.constants import Meta, MetaTombRaider, MetaSimPose
from openpilot.sunnypilot.modeld.runners import ModelRunner
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.system.hardware import PC
from openpilot.system.hardware.hw import Paths
from pathlib import Path

USE_ONNX = os.getenv('USE_ONNX', PC)

CUSTOM_MODEL_PATH = Paths.model_root()
METADATA_PATH = Path(__file__).parent / '../models/supercombo_metadata.pkl'

ModelManager = custom.ModelManagerSP


def get_model_path():
  if USE_ONNX:
    return {ModelRunner.ONNX: Path(__file__).parent / '../models/supercombo.onnx'}

  if bundle := get_active_bundle():
    drive_model = next(model for model in bundle.models if model.type == ModelManager.Type.drive)
    return {ModelRunner.THNEED: f"{CUSTOM_MODEL_PATH}/{drive_model.fileName}"}

  return {ModelRunner.THNEED: Path(__file__).parent / '../models/supercombo.thneed'}


def load_metadata():
  metadata_path = METADATA_PATH

  if bundle := get_active_bundle():
    metadata_model = next(model for model in bundle.models if model.type == ModelManager.Type.metadata)
    metadata_path = f"{CUSTOM_MODEL_PATH}/{metadata_model.fileName}"

  with open(metadata_path, 'rb') as f:
    return pickle.load(f)


def prepare_inputs(model_metadata) -> dict[str, np.ndarray]:
  # img buffers are managed in openCL transform code so we don't pass them as inputs
  inputs = {
    k: np.zeros(v, dtype=np.float32).flatten()
    for k, v in model_metadata['input_shapes'].items()
    if 'img' not in k
  }

  return inputs


def load_meta_constants(model_metadata):
  """
  Determines and loads the appropriate meta model class based on the metadata provided. The function checks
  specific keys and conditions within the provided metadata dictionary to identify the corresponding meta
  model class to return.

  :param model_metadata: Dictionary containing metadata about the model. It includes
      details such as input shapes, output slices, and other configurations for identifying
      metadata-dependent meta model classes.
  :type model_metadata: dict
  :return: The appropriate meta model class (Meta, MetaSimPose, or MetaTombRaider)
      based on the conditions and metadata provided.
  :rtype: type
  """
  meta = Meta  # Default Meta

  if 'sim_pose' in model_metadata['input_shapes'].keys():
    # Meta for models with sim_pose input
    meta = MetaSimPose
  else:
    # Meta for Tomb Raider, it does not include sim_pose input but has the same meta slice as previous models
    meta_slice = model_metadata['output_slices']['meta']
    meta_tf_slice = slice(5868, 5921, None)

    if (
          meta_slice.start == meta_tf_slice.start and
          meta_slice.stop == meta_tf_slice.stop and
          meta_slice.step == meta_tf_slice.step
    ):
      meta = MetaTombRaider

  return meta
