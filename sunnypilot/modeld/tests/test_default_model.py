from openpilot.sunnypilot.modeld.default_model import get_hash, MODEL_HASH_PATH, ONNX_PATH


class TestDefaultModel:
  def test_compare_onnx_hashes(self):
    new_hash = get_hash(ONNX_PATH)

    with open(MODEL_HASH_PATH) as f:
      current_hash = f.read().strip()

    assert new_hash == current_hash, "Run sunnypilot/modeld/default_model.py to update the default model name and hash"
