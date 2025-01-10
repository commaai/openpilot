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

def get_active_bundle(params: Params) -> custom.ModelManagerSP.ModelBundle:
  """Gets the active model bundle from cache"""
  if params is None:
    params = Params()

  if active_bundle := params.get("ModelManager_ActiveBundle"):
    return messaging.log_from_bytes(active_bundle, custom.ModelManagerSP.ModelBundle)

  return None
