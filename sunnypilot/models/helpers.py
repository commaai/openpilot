"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import hashlib
import os
import pickle
import numpy as np
import json

from openpilot.common.params import Params
from cereal import custom
from openpilot.sunnypilot.modeld.constants import Meta, MetaTombRaider, MetaSimPose
from openpilot.sunnypilot.modeld.runners import ModelRunner
from openpilot.system.hardware import PC
from openpilot.system.hardware.hw import Paths
from pathlib import Path

CURRENT_SELECTOR_VERSION = 3
REQUIRED_MIN_SELECTOR_VERSION = 2

USE_ONNX = os.getenv('USE_ONNX', PC)

CUSTOM_MODEL_PATH = Paths.model_root()
METADATA_PATH = Path(__file__).parent / '../models/supercombo_metadata.pkl'

ModelManager = custom.ModelManagerSP


async def verify_file(file_path: str, expected_hash: str) -> bool:
  """Verifies file hash against expected hash"""
  if not os.path.exists(file_path):
    return False

  sha256_hash = hashlib.sha256()
  with open(file_path, "rb") as file:
    for chunk in iter(lambda: file.read(4096), b""):
      sha256_hash.update(chunk)

  return sha256_hash.hexdigest().lower() == expected_hash.lower()


def is_bundle_version_compatible(bundle: dict) -> bool:
  """
  Checks whether the model bundle is compatible with the current selector version constraints.

  The bundle specifies a `minimum_selector_version`, which defines the minimum selector version
  required to load the model. This function ensures that:

    1. The model is not too old: the bundle must require at least `REQUIRED_MIN_SELECTOR_VERSION`.
    2. The model is not too new: it must support the current selector version (`CURRENT_SELECTOR_VERSION`).

  This allows the selector to enforce both a minimum and maximum range of supported models,
  even if a model would otherwise be compatible.

  :param bundle: Dictionary containing `minimum_selector_version`, as defined by the model bundle.
  :type bundle: Dict
  :return: True if the selector version is within the accepted range for the bundle; otherwise False.
  :rtype: Bool
  """
  return bool(REQUIRED_MIN_SELECTOR_VERSION <= bundle.get("minimumSelectorVersion", 0) <= CURRENT_SELECTOR_VERSION)

def get_active_bundle(params: Params = None) -> custom.ModelManagerSP.ModelBundle:
  """Gets the active model bundle from cache"""
  if params is None:
    params = Params()

  try:
    if (active_bundle := json.loads(params.get("ModelManager_ActiveBundle") or "{}")) and is_bundle_version_compatible(active_bundle):
      return custom.ModelManagerSP.ModelBundle(**active_bundle)
  except Exception:
    pass

  return None


def get_active_model_runner(params: Params = None, force_check=False) -> custom.ModelManagerSP.Runner:
  """
  Determines and returns the active model runner type, based on provided parameters.
  The function utilizes caching to prevent redundant calculations and checks.

  If the cached "ModelRunnerTypeCache" exists in the provided parameters and `force_check`
  is set to False, the cached value is directly returned. Otherwise, the function determines
  the runner type based on the active model bundle. If a model bundle containing a drive
  model exists, the runner type is derived based on the filename of the drive model.
  Finally, it updates the cache with the determined runner type, if needed.

  :param params: The parameter set used to retrieve caching and runner details. If `None`,
      a default `Params` instance is created internally.
  :type params: Params
  :param force_check: A flag indicating whether to bypass cached results and always
      re-determine the runner type. Defaults to `False`.
  :type force_check: bool
  :return: The determined or cached model runner type.
  :rtype: custom.ModelManagerSP.Runner
  """
  if params is None:
    params = Params()

  if (cached_runner_type := params.get("ModelRunnerTypeCache")) and not force_check:
    if isinstance(cached_runner_type, str) and cached_runner_type.isdigit():
      return int(cached_runner_type)

  runner_type = custom.ModelManagerSP.Runner.stock

  if active_bundle := get_active_bundle(params):
    runner_type = active_bundle.runner.raw

  if cached_runner_type != runner_type:
    params.put("ModelRunnerTypeCache", str(int(runner_type)))

  return runner_type

def _get_model():
  if bundle := get_active_bundle():
    drive_model = next(model for model in bundle.models if model.type == ModelManager.Model.Type.supercombo)
    return drive_model

  return None

def get_model_path():
  if USE_ONNX:
    return {ModelRunner.ONNX: Path(__file__).parent / '../models/supercombo.onnx'}

  if model := _get_model():
    return {ModelRunner.THNEED: f"{CUSTOM_MODEL_PATH}/{model.artifact.fileName}"}

  return {ModelRunner.THNEED: Path(__file__).parent / '../models/supercombo.thneed'}


def load_metadata():
  metadata_path = METADATA_PATH

  if model := _get_model():
    metadata_path = f"{CUSTOM_MODEL_PATH}/{model.metadata.fileName}"

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
