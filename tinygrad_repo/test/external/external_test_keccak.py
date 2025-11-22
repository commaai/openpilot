import unittest, zipfile, re
from tinygrad import Tensor
from tinygrad.helpers import fetch, tqdm

SHA3_URL = "https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/sha3/sha-3bytetestvectors.zip"
SHAKE_URL = "https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/sha3/shakebytetestvectors.zip"

class TestExternalKeccak(unittest.TestCase):
  def test_sha3_224(self): self.check_nist_vectors(SHA3_URL, ["SHA3_224LongMsg.rsp", "SHA3_224ShortMsg.rsp"], "sha3_224")
  def test_sha3_256(self): self.check_nist_vectors(SHA3_URL, ["SHA3_256LongMsg.rsp", "SHA3_256ShortMsg.rsp"], "sha3_256")
  def test_shake_128(self): self.check_nist_vectors(SHAKE_URL, ["SHAKE128LongMsg.rsp", "SHAKE128ShortMsg.rsp"], "shake_128")

  def check_nist_vectors(self, url: str, filenames: list[str], preset: str):
    pattern = r"Len\s*=\s*(?P<Len>\d+)\s+Msg\s*=\s*(?P<Msg>[0-9a-fA-F\s]+)\s+(MD|Output)\s*=\s*(?P<Output>[0-9a-fA-F]+)"
    vecs_zip = fetch(url)

    for filename in filenames:
      vecs = zipfile.ZipFile(vecs_zip).open(filename).read().decode()

      vectors = [ (l, bytes.fromhex(match["Msg"].lower()), bytes.fromhex(match["Output"].lower()))
        for match in re.finditer(pattern, vecs) if (l:=int(match["Len"])) < 8192 ]

      self.assertTrue(len(vectors) > 0)

      print("file", filename)
      for data_len, data, output in tqdm(vectors):
        tinyout = bytes(Tensor(data[:data_len//8]).keccak(preset).data())
        self.assertEqual(tinyout, output)

if __name__ == '__main__':
  unittest.main()
