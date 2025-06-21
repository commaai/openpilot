from typing_extensions import Callable
import hashlib, random, unittest
from tinygrad import Tensor, Device, getenv, dtypes
from tinygrad.device import is_dtype_supported

@unittest.skipUnless(is_dtype_supported(dtypes.uint8) and is_dtype_supported(dtypes.uint64), "Device must support uint8 and uint64")
@unittest.skipIf(getenv("MOCKGPU") and Device.DEFAULT == "NV", "crashes in NV CI")
class TestKeccak(unittest.TestCase):
  def setUp(self) -> None: random.seed(1337)

  def test_shape_keeping(self):
    s = (1, 2, 3, 4)
    for i in range(len(s)):
      out_shape = Tensor.randint(*s[i:], high=255, dtype=dtypes.uint8).keccak().shape
      self.assertTupleEqual(s[i:-1], out_shape[:-1])

  def test_sha3_224(self): self._test_preset("sha3_224", [143, 144])
  def test_sha3_256(self): self._test_preset("sha3_256", [135, 136])
  def test_shake_128(self): self._test_preset("shake_128", [167, 168], lambda d: hashlib.shake_128(d).digest(16))

  def _test_preset(self, name: str, special_sizes: list[int], hasher: Callable[[bytes], bytes] | None = None):
    def default_hasher(d: bytes) -> bytes: return getattr(hashlib, name)(d).digest()
    if hasher is None: hasher = default_hasher

    for n in (special_sizes + [special_sizes[0] - 1]):
      a, b = random.randbytes(n), random.randbytes(n)

      ha_ref, hb_ref = hasher(a), hasher(b)
      tres = Tensor.stack(*(Tensor(d) for d in (a, b))).keccak(name)
      ha, hb = tres[0].data(), tres[1].data()

      self.assertEqual(ha_ref, ha)
      self.assertEqual(ha_ref, Tensor(a).keccak(name).data())
      self.assertEqual(hb_ref, hb)

  def test_abc(self):
    # https://www.di-mgt.com.au/sha_testvectors.html
    out = Tensor(b"abc").keccak()
    self.assertEqual(bytes(out.tolist()), bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))

if __name__ == "__main__":
  unittest.main()
