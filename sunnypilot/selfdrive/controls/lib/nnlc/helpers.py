"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os
from difflib import SequenceMatcher

from opendbc.car import structs
from openpilot.common.basedir import BASEDIR

TORQUE_NN_MODEL_PATH = os.path.join(BASEDIR, "sunnypilot", "neural_network_data", "neural_network_lateral_control")


def similarity(s1: str, s2: str) -> float:
  return SequenceMatcher(None, s1, s2).ratio()


def get_nn_model_path(CP: structs.CarParams) -> tuple[str | None, str, bool]:
  exact_match = True
  car_fingerprint = CP.carFingerprint
  eps_fw = str(next((fw.fwVersion for fw in CP.carFw if fw.ecu == "eps"), ""))
  model_name = ""

  def check_nn_path(nn_candidate):
    _model_path = None
    _max_similarity = -1.0
    for f in os.listdir(TORQUE_NN_MODEL_PATH):
      if f.endswith(".json"):
        model = os.path.splitext(f)[0]
        similarity_score = similarity(model, nn_candidate)
        if similarity_score > _max_similarity:
          _max_similarity = similarity_score
          _model_path = os.path.join(TORQUE_NN_MODEL_PATH, f)
    return _model_path, _max_similarity

  if len(eps_fw) > 3:
    eps_fw = eps_fw.replace("\\", "")
    nn_candidate = f"{car_fingerprint} {eps_fw}"
    model_path, max_similarity = check_nn_path(nn_candidate)

    if model_path is not None and car_fingerprint in model_path and max_similarity >= 0.9:
      model_name = os.path.splitext(os.path.basename(model_path))[0]
      exact_match = max_similarity >= 0.99
      return model_path, model_name, exact_match

  nn_candidate = car_fingerprint
  model_path, max_similarity = check_nn_path(nn_candidate)

  if model_path is None or car_fingerprint not in model_path or max_similarity < 0.9:
    model_path = None

  if model_path is not None:
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    exact_match = max_similarity >= 0.99

  return model_path, model_name, exact_match
