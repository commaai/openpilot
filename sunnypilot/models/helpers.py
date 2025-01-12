# Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
#
# This file is part of sunnypilot and is licensed under the MIT License.
# See the LICENSE.md file in the root directory for more details.

import hashlib
import os
from openpilot.common.params import Params
from cereal import custom, messaging


async def verify_file(file_path: str, expected_hash: str) -> bool:
  """Verifies file hash against expected hash"""
  if not os.path.exists(file_path):
    return False

  sha256_hash = hashlib.sha256()
  with open(file_path, "rb") as file:
    for chunk in iter(lambda: file.read(4096), b""):
      sha256_hash.update(chunk)

  return sha256_hash.hexdigest().lower() == expected_hash.lower()


def get_active_bundle(params: Params = None) -> custom.ModelManagerSP.ModelBundle:
  """Gets the active model bundle from cache"""
  if params is None:
    params = Params()

  if active_bundle := params.get("ModelManager_ActiveBundle"):
    return messaging.log_from_bytes(active_bundle, custom.ModelManagerSP.ModelBundle)

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

  runner_type = custom.ModelManagerSP.Runner.tinygrad

  if active_bundle := get_active_bundle(params):
    runner_type = active_bundle.runner.raw

  if cached_runner_type != runner_type:
    params.put("ModelRunnerTypeCache", str(int(runner_type)))

  return runner_type
