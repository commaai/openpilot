import pickle, unittest
from collections import Counter
from pathlib import Path

from extra.nv_pma.decode import decode
from tinygrad.helpers import DEBUG

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLES_5090_DIR = Path(__file__).parent.parent / "examples_5090"

def decode_and_aggregate(raw_dumps: list[bytes], sm_version: int = 0x800) -> Counter[tuple[int, int]]:
  """Decode all PMA buffers and aggregate by (relative_pc, stall_reason). Each dump is normalized separately."""
  result: Counter[tuple[int, int]] = Counter()
  for raw in raw_dumps:
    samples = [s for s, _ in decode(raw, sm_version)]
    if not samples: continue
    base_pc = min(s.pc_offset for s in samples)
    result += Counter((s.pc_offset - base_pc, int(s.stall_reason)) for s in samples)
  return result

def cupti_to_counter(cupti_records: list[dict]) -> Counter[tuple[int, int]]:
  """Convert CUPTI records to Counter[(pcOffset, stallReason)]."""
  counter: Counter[tuple[int, int]] = Counter()
  for r in cupti_records:
    counter[(r['pcOffset'], r['stallReason'])] += r['samples']
  return counter

class TestNVProf(unittest.TestCase):
  def _test_example(self, name: str, sm_version: int = 0x800, examples_dir: Path = EXAMPLES_DIR):
    pkl_file = examples_dir / f"{name}.pkl"
    if not pkl_file.exists():
      self.skipTest(f"Example data not found: {pkl_file}. Run collect.py first.")

    with open(pkl_file, "rb") as f:
      data = pickle.load(f)

    self.assertEqual(data["test_name"], name)
    pma_agg = decode_and_aggregate(data["pma_raw_dumps"], sm_version)
    cupti_agg = cupti_to_counter(data["cupti_pc_samples"])

    if DEBUG >= 2:
      total = sum(cupti_agg.values())
      mismatched = sum(abs(pma_agg.get(k, 0) - v) for k, v in cupti_agg.items())
      mismatched += sum(v for k, v in pma_agg.items() if k not in cupti_agg)
      mismatched //= 2

      print(f"\n=== Test: {name} ===")
      print(f"Total samples: {total}, Mismatched: {mismatched} ({mismatched/total*100 if total else 0:.1f}%)")

    self.assertEqual(pma_agg, cupti_agg, f"PMA: {dict(pma_agg)}\nCUPTI: {dict(cupti_agg)}")

  # Ampere tests (8-byte format)
  def test_decode_test_plus(self): self._test_example("test_plus")
  def test_decode_test_reduce_sum(self): self._test_example("test_reduce_sum")
  def test_decode_test_broadcast(self): self._test_example("test_broadcast")
  def test_decode_test_matmul(self): self._test_example("test_matmul")
  def test_decode_test_plus_big(self): self._test_example("test_plus_big")
  def test_decode_test_elementwise_chain(self): self._test_example("test_elementwise_chain")
  def test_decode_test_conv2d(self): self._test_example("test_conv2d")
  def test_decode_test_large_matmul(self): self._test_example("test_large_matmul")

  # Blackwell/5090 tests (9-byte format)
  def test_5090_test_plus(self): self._test_example("test_plus", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_plus_big(self): self._test_example("test_plus_big", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_broadcast(self): self._test_example("test_broadcast", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_matmul(self): self._test_example("test_matmul", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_large_matmul(self): self._test_example("test_large_matmul", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_reduce_sum(self): self._test_example("test_reduce_sum", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_reduce_max(self): self._test_example("test_reduce_max", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_elementwise_chain(self): self._test_example("test_elementwise_chain", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_conv2d(self): self._test_example("test_conv2d", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_exp(self): self._test_example("test_exp", 0xa04, EXAMPLES_5090_DIR)
  def test_5090_test_softmax(self): self._test_example("test_softmax", 0xa04, EXAMPLES_5090_DIR)

if __name__ == "__main__":
  unittest.main()
