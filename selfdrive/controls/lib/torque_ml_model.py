import pickle
import numpy as np


def _leaky_relu(x: np.ndarray, negative_slope: float = 0.3) -> np.ndarray:
  return np.where(x > 0, x, x * negative_slope)


class TorqueMLModel:
  """
  Minimal Konverter-style Dense+LeakyReLU(0.3) inference.

  Expects:
  - weights.npz containing wb = [w_list, b_list]
  - norm.pkl containing feature_names/x_center/x_scale/y_center/y_scale
  """

  def __init__(self, weights_npz: str, norm_pkl: str):
    wb = np.load(weights_npz, allow_pickle=True)
    self.w, self.b = wb["wb"]

    with open(norm_pkl, "rb") as f:
      payload = pickle.load(f)

    self.feature_names = list(payload["feature_names"])
    self.x_center = np.asarray(payload["x_center"], dtype=np.float32)
    self.x_scale = np.asarray(payload["x_scale"], dtype=np.float32)
    self.y_center = float(payload["y_center"])
    self.y_scale = float(payload["y_scale"])

    # Sanity
    if len(self.feature_names) != len(self.x_center) or len(self.x_center) != len(self.x_scale):
      raise ValueError("Normalization shape mismatch")

  def predict(self, features: list[float]) -> float:
    x = np.asarray(features, dtype=np.float32).reshape((1, -1))
    x = (x - self.x_center) / self.x_scale

    l = x
    # Dense, LeakyReLU, Dense, LeakyReLU, Dense(1)
    l = _leaky_relu(l @ self.w[0] + self.b[0])
    l = _leaky_relu(l @ self.w[1] + self.b[1])
    y = l @ self.w[2] + self.b[2]
    return float(y.reshape(-1)[0] * self.y_scale + self.y_center)

