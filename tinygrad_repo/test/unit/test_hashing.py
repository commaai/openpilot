from typing_extensions import Callable
import hashlib, random, unittest
from tinygrad import Tensor, Device, getenv, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import CI

@unittest.skipUnless(is_dtype_supported(dtypes.uint8) and is_dtype_supported(dtypes.uint64), "Device must support uint8 and uint64")
@unittest.skipIf(getenv("MOCKGPU") and Device.DEFAULT == "NV", "crashes in NV CI")
class TestHashing(unittest.TestCase):
  def _python_hash_1mb(self, data:bytes):
    chunks = [data[i:i+4096] for i in range(0, len(data), 4096)]
    chunk_hashes = [hashlib.shake_128(chunk).digest(16) for chunk in chunks]
    return hashlib.shake_128(b''.join(chunk_hashes)).digest(16)

  @unittest.skipIf(CI, "very slow")
  def test_abc(self):
    expected = self._python_hash_1mb(b"abc" + b"\x00" * (2**20 - 3))
    out = Tensor(b"abc").hash()
    self.assertEqual(bytes(out.data()), expected)

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

  def test_referenced(self):
    # https://www.di-mgt.com.au/sha_testvectors.html
    self.assertEqual(bytes(Tensor(b"abc").keccak().tolist()),
                     bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))
    self.assertEqual(bytes(Tensor(b"").keccak().tolist()),
                     bytearray.fromhex("a7ffc6f8bf1ed766 51c14756a061d662 f580ff4de43b49fa 82d80a4b80f8434a"))
    t = Tensor(b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu").keccak()
    self.assertEqual(bytes(t.tolist()),
                     bytearray.fromhex("916f6061fe879741 ca6469b43971dfdb 28b1a32dc36cb325 4e812be27aad1d18"))
    # TODO: this does not run or very slow
    # self.assertEqual(bytes(Tensor(b"a" * 1000000).keccak().tolist()),
    #                  bytearray.fromhex("5c8875ae474a3634 ba4fd55ec85bffd6 61f32aca75c6d699 d0cdcb6c115891c1"))

  def test_long(self):
    data = b"\x00" * 4
    self.assertEqual(bytes(Tensor(data).keccak("shake_128").tolist()), hashlib.shake_128(data).digest(16))

    data = b"\x00" * (1000 if CI else 4096)
    self.assertEqual(bytes(Tensor(data).keccak("shake_128").tolist()), hashlib.shake_128(data).digest(16))

if __name__ == "__main__":
  unittest.main()
