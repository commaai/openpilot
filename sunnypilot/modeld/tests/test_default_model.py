import os
import hashlib

from openpilot.common.basedir import BASEDIR

MODEL_HASH_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDefaultModel:
  @classmethod
  def setup_class(cls):
    cls.onnx_path = os.path.join(BASEDIR, "selfdrive", "modeld", "models", "supercombo.onnx")
    cls.current_hash_path = os.path.join(MODEL_HASH_DIR, "model_hash")

  @staticmethod
  def get_hash(path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
      for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

  def test_compare_onnx_hashes(self):
    new_hash = self.get_hash(str(self.onnx_path))

    with open(self.current_hash_path) as f:
      current_hash = f.read().strip()

    assert new_hash == current_hash, (
      "Driving model updated!\n" +
      f"Current hash: {current_hash}\n" +
      f"New hash: {new_hash}\n" +
      "Please update common/model.h if the default driving model name has changed."
    )
