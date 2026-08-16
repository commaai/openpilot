import unittest, time
from tinygrad import Tensor

class TestScheduleScaling(unittest.TestCase):
  """Test that .schedule() scales linearly with graph size (no O(n^2) behavior)."""

  def _assert_linear(self, fn, n_small=200, n_large=1000):
    """Assert schedule time scales at most ~linearly: time(n_large)/time(n_small) should be close to n_large/n_small."""
    fn(n_small).schedule_linear()  # warmup
    t_small = min(self._time_schedule(fn, n) for n in [n_small]*3)
    t_large = min(self._time_schedule(fn, n) for n in [n_large]*3)
    size_ratio = n_large / n_small  # 5.0
    time_ratio = t_large / t_small
    # O(n) -> time_ratio ~ 5, O(n^2) -> time_ratio ~ 25. threshold at 10 catches n^2 with margin.
    self.assertLess(time_ratio / size_ratio, 2.0,
      f"schedule appears superlinear: n={n_small} {t_small*1e3:.1f}ms, n={n_large} {t_large*1e3:.1f}ms "
      f"(time grew {time_ratio:.1f}x for {size_ratio:.0f}x size, per-node ratio {time_ratio/size_ratio:.2f})")

  @staticmethod
  def _time_schedule(fn, n) -> float:
    st = time.perf_counter()
    fn(n).schedule_linear()
    return time.perf_counter() - st

  # *** rangeify: ending_ranges accumulation and consumer merge ***

  # ending_ranges accumulation via sum([], []) and nested scan in run_rangeify.
  # this creates reduce ops whose ending_ranges lists grow with graph depth, causing O(n^2) list copies.
  def test_multi_reduce_scaling(self):
    def multi_reduce(n):
      x = Tensor.empty(256, 256)
      for _ in range(n):
        s = x.sum(axis=-1, keepdim=True)
        x = x + s + s
      return x
    self._assert_linear(multi_reduce)

  # reduce+elementwise chain stresses ending_ranges propagation and post-rangeify rewrites
  def test_wide_reduce_scaling(self):
    def wide_reduce(n):
      x = Tensor.empty(256, 256)
      for _ in range(n):
        x = x + x.sum(axis=-1, keepdim=True)
      return x
    self._assert_linear(wide_reduce)

  # expand ops inject into ending_ranges via the EXPAND path in run_rangeify
  def test_expand_reduce_scaling(self):
    def expand_reduce(n):
      x = Tensor.empty(256, 1)
      for _ in range(n):
        y = x.expand(256, 256)
        x = (y + y).sum(axis=-1, keepdim=True)
      return x
    self._assert_linear(expand_reduce)

  # *** graph_rewrite: multi-consumer DAG patterns ***

  # multi-consumer diamond pattern (fan-out/fan-in) stresses consumer_rngs merge in run_rangeify
  def test_diamond_scaling(self):
    def diamond(n):
      x = Tensor.empty(256, 256)
      for _ in range(n):
        a = x + 1
        b = x + 2
        x = a + b
      return x
    self._assert_linear(diamond)

  # elementwise chain baseline — should be trivially O(n)
  def test_chain_scaling(self):
    def chain(n):
      x = Tensor.empty(256, 256)
      for _ in range(n): x = x + 1
      return x
    self._assert_linear(chain)

  # softmax has multi-consumer structure (x used for max, exp, and sum), stresses graph_rewrite on DAGs
  def test_softmax_scaling(self):
    def softmax_chain(n):
      x = Tensor.empty(64, 256)
      for _ in range(n): x = x.softmax(axis=-1)
      return x
    self._assert_linear(softmax_chain)

  # *** post-rangeify: symbolic rewrites, kernel splitting ***

  # matmul chain stresses symbolic+reduce_collapse and split_store
  def test_matmul_scaling(self):
    def matmul_chain(n):
      xs = [Tensor.empty(32, 32) for _ in range(n + 1)]
      result = xs[0]
      for i in range(n): result = result @ xs[i + 1]
      return result
    self._assert_linear(matmul_chain)

  # contiguous chain stresses remove_bufferize callbacks (toposort per BUFFERIZE node)
  def test_contiguous_scaling(self):
    def contiguous_chain(n):
      x = Tensor.empty(256, 256)
      for _ in range(n): x = (x + 1).contiguous()
      return x
    self._assert_linear(contiguous_chain)

  # *** schedule: AFTER handling, assign ***

  # assign chain stresses AFTER cycle detection (toposort inside toposort loop in get_rangeify_map)
  def test_assign_scaling(self):
    def assign_chain(n):
      x = Tensor.empty(256, 256).realize()
      for _ in range(n): x.assign(x + 1)
      return x
    self._assert_linear(assign_chain)

  # layernorm has multi-consumer reduces (mean reused in variance), stresses consumer_rngs merge and symbolic rewrites
  def test_layernorm_scaling(self):
    def layernorm_chain(n):
      x = Tensor.empty(64, 256)
      for _ in range(n):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        x = (x - mean) / (var + 1e-5).sqrt()
      return x
    self._assert_linear(layernorm_chain)

  # concat chain stresses MSTACK/MSELECT handling and wide SINK construction
  def test_concat_scaling(self):
    def concat_chain(n):
      parts = [Tensor.empty(4, 256) + i for i in range(n)]
      return parts[0].cat(*parts[1:])
    self._assert_linear(concat_chain)

if __name__ == '__main__':
  unittest.main(verbosity=2)
