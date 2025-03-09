from openpilot.sunnypilot.models.default_model import get_file_hash, MODEL_HASH_PATH, VISION_ONNX_PATH, POLICY_ONNX_PATH
import hashlib


class TestDefaultModel:
  def test_compare_onnx_hashes(self):
    vision_hash = get_file_hash(VISION_ONNX_PATH)
    policy_hash = get_file_hash(POLICY_ONNX_PATH)

    combined_hash = hashlib.sha256((vision_hash + policy_hash).encode()).hexdigest()

    with open(MODEL_HASH_PATH) as f:
      current_hash = f.read().strip()

    assert combined_hash == current_hash, "Run sunnypilot/models/default_model.py to update the default model name and hash"
