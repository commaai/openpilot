import numpy as np
import pytest

from openpilot.selfdrive.modeld.parse_model_outputs import safe_exp, sigmoid, softmax, Parser
from openpilot.selfdrive.modeld.constants import ModelConstants


# ---------------------------------------------------------------------------
# safe_exp
# ---------------------------------------------------------------------------

class TestSafeExp:
  def test_normal_values(self):
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(safe_exp(x), np.exp(x), rtol=1e-6)

  def test_clips_large_positive(self):
    x = np.array([100.0, 50.0, 20.0], dtype=np.float32)
    result = safe_exp(x)
    expected_cap = np.exp(11.0)
    np.testing.assert_allclose(result, [expected_cap] * 3, rtol=1e-6)

  def test_large_negative(self):
    x = np.array([-100.0, -50.0], dtype=np.float32)
    result = safe_exp(x)
    assert np.all(result >= 0.0)
    assert np.all(result < 1e-10)

  def test_scalar(self):
    assert float(safe_exp(np.float32(0.0))) == pytest.approx(1.0)

  def test_float16_no_overflow(self):
    x = np.array([100.0, 11.0, -5.0], dtype=np.float16)
    result = safe_exp(x)
    assert np.all(np.isfinite(result))

  def test_out_parameter(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.empty_like(x)
    result = safe_exp(x, out=out)
    np.testing.assert_array_equal(result, out)

  def test_zero(self):
    assert float(safe_exp(np.float64(0.0))) == pytest.approx(1.0)

  def test_boundary_at_11(self):
    below = safe_exp(np.float64(10.9))
    at = safe_exp(np.float64(11.0))
    above = safe_exp(np.float64(11.1))
    assert float(at) == pytest.approx(float(above), rel=1e-6)
    assert float(below) < float(at)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
  def test_zero(self):
    assert float(sigmoid(np.float64(0.0))) == pytest.approx(0.5)

  def test_large_positive(self):
    result = float(sigmoid(np.float64(100.0)))
    assert result == pytest.approx(1.0, abs=1e-6)

  def test_large_negative(self):
    result = float(sigmoid(np.float64(-100.0)))
    assert result == pytest.approx(0.0, abs=1e-4)

  def test_output_range(self):
    x = np.linspace(-10, 10, 100, dtype=np.float32)
    result = sigmoid(x)
    assert np.all(result > 0.0)
    assert np.all(result < 1.0)

  def test_monotonically_increasing(self):
    x = np.linspace(-10, 10, 100, dtype=np.float64)
    result = sigmoid(x)
    assert np.all(np.diff(result) > 0)

  def test_symmetry(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert float(sigmoid(x[0]) + sigmoid(-x[0])) == pytest.approx(1.0)
    assert float(sigmoid(x[1]) + sigmoid(-x[1])) == pytest.approx(1.0)
    assert float(sigmoid(x[2]) + sigmoid(-x[2])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
  def test_sums_to_one(self):
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = softmax(x.copy())
    np.testing.assert_allclose(result.sum(axis=-1), [1.0], atol=1e-6)

  def test_non_negative(self):
    x = np.array([[-5.0, 0.0, 5.0]], dtype=np.float32)
    result = softmax(x.copy())
    assert np.all(result >= 0.0)

  def test_uniform_input(self):
    x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    result = softmax(x.copy())
    np.testing.assert_allclose(result, [[0.25, 0.25, 0.25, 0.25]], atol=1e-6)

  def test_numerically_stable_large_values(self):
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float64)
    result = softmax(x.copy())
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result.sum(axis=-1), [1.0], atol=1e-6)

  def test_float32_in_place(self):
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = softmax(x)
    np.testing.assert_allclose(result.sum(axis=-1), [1.0], atol=1e-6)

  def test_float16(self):
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
    result = softmax(x.copy())
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(float(result.sum()), 1.0, atol=1e-2)

  def test_custom_axis(self):
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = softmax(x.copy(), axis=0)
    np.testing.assert_allclose(result.sum(axis=0), [1.0, 1.0], atol=1e-6)

  def test_batch(self):
    x = np.random.default_rng(42).standard_normal((4, 8)).astype(np.float32)
    result = softmax(x.copy())
    np.testing.assert_allclose(result.sum(axis=-1), np.ones(4), atol=1e-6)
    assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# Parser.check_missing
# ---------------------------------------------------------------------------

class TestParserCheckMissing:
  def test_raises_when_missing_and_not_ignored(self):
    parser = Parser(ignore_missing=False)
    with pytest.raises(ValueError, match="Missing output"):
      parser.check_missing({}, "nonexistent")

  def test_returns_true_when_missing_and_ignored(self):
    parser = Parser(ignore_missing=True)
    assert parser.check_missing({}, "nonexistent") is True

  def test_returns_false_when_present(self):
    parser = Parser(ignore_missing=False)
    outs = {"key": np.array([1.0])}
    assert parser.check_missing(outs, "key") is False


# ---------------------------------------------------------------------------
# Parser.parse_binary_crossentropy
# ---------------------------------------------------------------------------

class TestParseBinaryCrossentropy:
  def test_applies_sigmoid(self):
    parser = Parser()
    raw = np.array([[0.0, 2.0, -2.0]], dtype=np.float32)
    outs = {"test": raw.copy()}
    parser.parse_binary_crossentropy("test", outs)
    expected = sigmoid(raw)
    np.testing.assert_allclose(outs["test"], expected, rtol=1e-5)

  def test_output_in_zero_one(self):
    parser = Parser()
    raw = np.random.default_rng(42).standard_normal((2, 10)).astype(np.float32)
    outs = {"test": raw}
    parser.parse_binary_crossentropy("test", outs)
    assert np.all(outs["test"] > 0.0)
    assert np.all(outs["test"] < 1.0)

  def test_skips_missing_with_ignore(self):
    parser = Parser(ignore_missing=True)
    outs = {}
    parser.parse_binary_crossentropy("absent", outs)
    assert "absent" not in outs


# ---------------------------------------------------------------------------
# Parser.parse_categorical_crossentropy
# ---------------------------------------------------------------------------

class TestParseCategoricalCrossentropy:
  def test_applies_softmax(self):
    parser = Parser()
    raw = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    outs = {"test": raw.copy()}
    parser.parse_categorical_crossentropy("test", outs)
    result = outs["test"]
    np.testing.assert_allclose(result.sum(axis=-1), [1.0], atol=1e-6)
    assert np.all(result >= 0.0)

  def test_reshape_with_out_shape(self):
    parser = Parser()
    batch = 2
    raw = np.random.default_rng(42).standard_normal((batch, 4 * 8)).astype(np.float32)
    outs = {"test": raw.copy()}
    parser.parse_categorical_crossentropy("test", outs, out_shape=(4, 8))
    assert outs["test"].shape == (batch, 4, 8)
    np.testing.assert_allclose(outs["test"].sum(axis=-1), np.ones((batch, 4)), atol=1e-5)

  def test_skips_missing_with_ignore(self):
    parser = Parser(ignore_missing=True)
    outs = {}
    parser.parse_categorical_crossentropy("absent", outs)
    assert "absent" not in outs


# ---------------------------------------------------------------------------
# Parser.parse_mdn (simple in_N=0 path)
# ---------------------------------------------------------------------------

class TestParseMdn:
  def test_basic_shape_in_N_0(self):
    parser = Parser()
    batch = 2
    out_shape = (ModelConstants.POSE_WIDTH,)
    n_values = ModelConstants.POSE_WIDTH
    raw_width = n_values * 2
    raw = np.random.default_rng(42).standard_normal((batch, raw_width)).astype(np.float32)
    outs = {"pose": raw.copy()}
    parser.parse_mdn("pose", outs, in_N=0, out_N=0, out_shape=out_shape)
    assert outs["pose"].shape == (batch,) + out_shape
    assert outs["pose_stds"].shape == (batch,) + out_shape

  def test_stds_are_positive(self):
    parser = Parser()
    batch = 3
    out_shape = (ModelConstants.POSE_WIDTH,)
    n_values = ModelConstants.POSE_WIDTH
    raw_width = n_values * 2
    raw = np.random.default_rng(99).standard_normal((batch, raw_width)).astype(np.float32)
    outs = {"pose": raw.copy()}
    parser.parse_mdn("pose", outs, in_N=0, out_N=0, out_shape=out_shape)
    assert np.all(outs["pose_stds"] > 0.0)

  def test_skips_missing_with_ignore(self):
    parser = Parser(ignore_missing=True)
    outs = {}
    parser.parse_mdn("absent", outs, in_N=0, out_N=0, out_shape=(6,))
    assert "absent" not in outs

  def test_mu_matches_raw_first_half(self):
    parser = Parser()
    batch = 1
    n_values = 4
    out_shape = (n_values,)
    mu_raw = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    std_raw = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    raw = np.concatenate([mu_raw, std_raw], axis=-1)
    outs = {"test": raw.copy()}
    parser.parse_mdn("test", outs, in_N=0, out_N=0, out_shape=out_shape)
    np.testing.assert_allclose(outs["test"], mu_raw, rtol=1e-6)

  def test_mdn_with_hypotheses(self):
    parser = Parser()
    batch = 1
    in_N = ModelConstants.PLAN_MHP_N
    out_N = ModelConstants.PLAN_MHP_SELECTION
    plan_width = ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH
    n_values = plan_width
    raw_per_hyp = n_values * 2 + out_N
    raw = np.random.default_rng(42).standard_normal((batch, in_N * raw_per_hyp)).astype(np.float32)
    outs = {"plan": raw.copy()}
    parser.parse_mdn("plan", outs, in_N=in_N, out_N=out_N,
                     out_shape=(ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
    assert outs["plan"].shape == (batch, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert outs["plan_stds"].shape == (batch, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert np.all(outs["plan_stds"] > 0.0)

def test_softmax_float32_is_inplace_and_normalized():
  raw = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
  out = softmax(raw, axis=-1)

  assert out is raw
  np.testing.assert_allclose(np.sum(out, axis=-1), np.array([1.0], dtype=np.float32))
  assert out[0, 2] > out[0, 1] > out[0, 0]


def test_softmax_float16_returns_new_array_and_normalized():
  raw = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
  out = softmax(raw, axis=-1)

  assert out is not raw
  np.testing.assert_allclose(np.sum(out, axis=-1), np.array([1.0], dtype=np.float16), atol=1e-3)
  assert out[0, 2] > out[0, 1] > out[0, 0]


def test_sigmoid_handles_extreme_values():
  vals = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
  out = sigmoid(vals)

  assert np.isfinite(out).all()
  assert out[0] < 1e-4
  assert out[1] == pytest.approx(0.5)
  assert out[2] > 1.0 - 1e-4


def test_missing_output_raises_by_default():
  parser = Parser(ignore_missing=False)
  outs = {}

  with pytest.raises(ValueError, match="Missing output missing"):
    parser.parse_binary_crossentropy("missing", outs)


def test_missing_output_is_ignored_when_configured():
  parser = Parser(ignore_missing=True)
  outs = {}

  parser.parse_binary_crossentropy("missing", outs)
  assert outs == {}


def test_parse_categorical_crossentropy_reshapes_and_normalizes():
  parser = Parser()
  outs = {"desire_state": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)}

  parser.parse_categorical_crossentropy("desire_state", outs, out_shape=(2, 2))

  parsed = outs["desire_state"]
  assert parsed.shape == (1, 2, 2)
  np.testing.assert_allclose(np.sum(parsed, axis=-1), np.ones((1, 2), dtype=np.float32))


def test_parse_mdn_single_output_selects_highest_weight_hypothesis():
  parser = Parser()
  # shape before parse: (batch, in_N * (2 * n_values + out_N))
  # here: batch=1, in_N=3, out_N=1, n_values=2 => 1 x (3 * 5) = 1 x 15
  raw = np.array([[
    10.0, 11.0, 0.0, 0.0, -2.0,   # hyp 0, low weight
    20.0, 21.0, 0.0, 0.0, 6.0,    # hyp 1, highest weight
    30.0, 31.0, 0.0, 0.0, 1.0,    # hyp 2, medium weight
  ]], dtype=np.float32)
  outs = {"plan": raw}

  parser.parse_mdn("plan", outs, in_N=3, out_N=1, out_shape=(2,))

  parsed = outs["plan"]
  assert parsed.shape == (1, 2)
  np.testing.assert_allclose(parsed[0], np.array([20.0, 21.0], dtype=np.float32))

  weights = outs["plan_weights"]
  assert weights.shape == (1, 3, 1)
  assert weights[0, 0, 0] >= weights[0, 1, 0] >= weights[0, 2, 0]
