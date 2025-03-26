"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os
import tomllib
from difflib import SequenceMatcher

from opendbc.car import structs
from openpilot.common.basedir import BASEDIR

TORQUE_NN_MODEL_PATH = os.path.join(BASEDIR, "sunnypilot", "neural_network_data", "neural_network_lateral_control")
TORQUE_NN_MODEL_SUBSTITUTE_PATH = os.path.join(BASEDIR, "opendbc", "car", "torque_data/substitute.toml")
MOCK_MODEL_PATH = os.path.join(TORQUE_NN_MODEL_PATH, "MOCK.json")


def similarity(s1: str, s2: str) -> float:
  return SequenceMatcher(None, s1, s2).ratio()


def get_nn_model_path(CP: structs.CarParams) -> tuple[str, str, bool]:
  car_fingerprint = CP.carFingerprint
  eps_fw = str(next((fw.fwVersion for fw in CP.carFw if fw.ecu == "eps"), ""))

  def check_nn_path(_nn_candidate):
    _model_path = None
    _max_similarity = -1.0
    for f in os.listdir(TORQUE_NN_MODEL_PATH):
      if f.endswith(".json"):
        model = os.path.splitext(f)[0]
        similarity_score = similarity(model, _nn_candidate)
        if similarity_score > _max_similarity:
          _max_similarity = similarity_score
          _model_path = os.path.join(TORQUE_NN_MODEL_PATH, f)
    return _model_path, _max_similarity

  if len(eps_fw) > 3:
    eps_fw = eps_fw.replace("\\", "")
    nn_candidate = f"{car_fingerprint} {eps_fw}"
  else:
    nn_candidate = car_fingerprint

  model_path, max_similarity = check_nn_path(nn_candidate)
  exact_match = max_similarity >= 0.99

  if car_fingerprint not in model_path or 0.0 <= max_similarity < 0.9:
    nn_candidate = car_fingerprint
    model_path, max_similarity = check_nn_path(nn_candidate)
    exact_match = max_similarity >= 0.99

    if 0.0 <= max_similarity < 0.9:
      with open(TORQUE_NN_MODEL_SUBSTITUTE_PATH, 'rb') as f:
        sub = tomllib.load(f)
      sub_candidate = sub.get(car_fingerprint, car_fingerprint)

      for candidate in [car_fingerprint, sub_candidate]:
        model_path, max_similarity = check_nn_path(candidate)

      exact_match = False

  if CP.steerControlType == structs.CarParams.SteerControlType.angle:
    model_path = MOCK_MODEL_PATH

  model_name = os.path.splitext(os.path.basename(model_path))[0]

  return model_path, model_name, exact_match
