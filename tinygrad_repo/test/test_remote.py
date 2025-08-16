import numpy as np, unittest, string
from hypothesis import given, strategies as st
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.runtime.ops_remote import RemoteDevice, parse_hosts
from tinygrad.runtime.graph.remote import RemoteGraph
from tinygrad.helpers import LazySeq, all_same, Context

def multihost_env(devices):
  def same_hosts(devices): return all_same([h for h,_ in devices])
  return isinstance(devices, list) and len(devices) >= 12 and not same_hosts(devices[0:12]) and same_hosts(devices[0:6]) and same_hosts(devices[6:12])

@unittest.skipUnless(Device.DEFAULT == "REMOTE" and multihost_env(RemoteDevice.devices), "Requires special environment")
class TestRemoteMultiHost(unittest.TestCase):
  def test_mutlihost_transfer(self):
    a = Tensor.arange(0, 16, device='REMOTE:0').contiguous().realize()
    b = a.to('REMOTE:6').contiguous().realize()
    np.testing.assert_equal(b.numpy(), np.arange(0, 16))

  @Context(JIT_BATCH_SIZE=2**32)
  def test_multihost_matmul_jit_graph(self):
    @TinyJit
    def do(a:Tensor, b:Tensor): return (a @ b).contiguous().realize()

    ds = ('REMOTE:0', 'REMOTE:1', 'REMOTE:6', 'REMOTE:7')
    for _ in range(3):
      na, nb = np.random.rand(128, 128).astype(np.float32), np.random.rand(128, 128).astype(np.float32)
      a, b = Tensor(na).shard(ds, 0).contiguous().realize(), Tensor(nb).shard(ds, 0).contiguous().realize()
      nc = na @ nb
      c = do(a, b)
      np.testing.assert_allclose(nc, c.numpy(), rtol=3e-2, atol=1e-4) # tolerances from extra/gemm/simple_matmul.py

    # Verify that everything is in one big cross-host graph
    assert len(do.captured._jit_cache) == 1 and isinstance(do.captured._jit_cache[0].prg, RemoteGraph), repr(do.captured)

  @Context(JIT_BATCH_SIZE=2**32)
  def test_multihost_aware_schedule(self):
    @TinyJit
    def do(*ts:Tensor):
      acc = Tensor.zeros(1, dtype=dtypes.float32)
      for t in ts: acc += t.sum()
      return acc.realize()

    def do_np(*ts:np.ndarray):
      acc = np.zeros(1, np.float32)
      for t in ts: acc += t.sum()
      return acc

    ds = ('REMOTE:0', 'REMOTE:1', 'REMOTE:6', 'REMOTE:7')
    TS = 64
    for _ in range(3):
      inp_np = [np.random.rand(256).astype(np.float32) for _ in range(TS)]
      inp = [Tensor(inp).shard(ds, 0).contiguous().realize() for inp in inp_np]
      out_np = do_np(*inp_np)
      out = do(*inp)
      np.testing.assert_allclose(out_np, out.numpy(), rtol=3e-2, atol=1e-4)

    # Verify that everything is in one big cross-host graph and that the scheduling is reasonable
    assert len(do.captured._jit_cache) == 1 and isinstance(do.captured._jit_cache[0].prg, RemoteGraph), repr(do.captured)
    # At the time of writing this: 2050 graph breaks without multihost aware scheduling, 14 with it. I've set fail threshold to 28 to not fail on
    # unrelated scheduling changes. Maybe 2x is a bit too pessimistic, but remote should perform just fine as long as this is not like a half hundred
    # or more here.
    self.assertLess(len(do.captured._jit_cache[0].prg.template), 28, "Very bad scheduling! Many unnecesary graph breaks!")

class TestParseHosts(unittest.TestCase):
  def assert_seq(self, result:LazySeq, host:str):
    self.assertIsInstance(result, LazySeq)
    for i in [0, 1, 5, 10]: self.assertEqual(result[i], (host, i))

  @given(st.sampled_from(["", "localhost", "192.168.1.1:8080", "host"]))
  def test_single_host_no_count(self, host:str):
    self.assert_seq(parse_hosts(host), host)

  @given(host=st.sampled_from(["localhost", "host", "192.168.1.1:8080"]), count=st.integers(0, 10))
  def test_single_host_with_count(self, host:str, count:int):
    self.assertEqual(parse_hosts(f"{host}*{count}"), [(host, i) for i in range(count)])

  def test_multiple_hosts_with_counts_simple(self):
    self.assertEqual(parse_hosts("host1*2,host2*3"), [("host1", i) for i in range(2)] + [("host2", i) for i in range(3)])

  @given(st.lists(st.tuples(st.text(alphabet=string.ascii_letters + string.digits + ".-:"), st.integers(1, 16)), min_size=1))
  def test_multiple_hosts_with_counts_sampled(self, host_count_pairs):
    hosts_str = ",".join(f"{host}*{count}" for host, count in host_count_pairs)
    expected = [(host, i) for host, count in host_count_pairs for i in range(count)]
    self.assertEqual(parse_hosts(hosts_str), expected)

  @given(st.sampled_from(["host1*2,host2", "a*1,b", "x*3,y*2,z"]))
  def test_mixed_hosts_fails(self, hosts):
    with self.assertRaises(AssertionError): parse_hosts(hosts)

  @given(st.sampled_from(["host*abc", "test*xyz", "a*1.5"]))
  def test_invalid_count_fails(self, hosts):
    with self.assertRaises(ValueError): parse_hosts(hosts)

  @given(st.sampled_from(["host*2*3", "a*1*2*3", "test*x*y"]))
  def test_multiple_asterisks_fails(self, hosts):
    with self.assertRaises(ValueError): parse_hosts(hosts)

if __name__ == '__main__':
  unittest.main()
