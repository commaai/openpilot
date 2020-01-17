import sys
import json
# pip2 install msgpack-python
import msgpack
import zlib
import os
import logging

from Crypto.Cipher import AES

ext = ".gz"
SWAG = '\xde\xe2\x11\x15VVC\xf2\x8ep\xd7\xe4\x87\x8d,9'

def compress_json(in_file, out_file):
  logging.debug("compressing %s -> %s", in_file, out_file)

  errors = 0

  good = []
  last_can_time = 0
  with open(in_file, 'r') as inf:
    for ln in inf:
      ln = ln.rstrip()
      if not ln: continue
      try:
        ll = json.loads(ln)
      except ValueError:
        errors += 1
        continue
      if ll is None or ll[0] is None:
        continue
      if ll[0][1] == 1:
        # no CAN in hex
        ll[1][2] = ll[1][2].decode("hex")
        # relativize the CAN timestamps
        this_can_time = ll[1][1]
        ll[1] = [ll[1][0], this_can_time - last_can_time, ll[1][2]]
        last_can_time = this_can_time
      good.append(ll)

  logging.debug("compressing %s -> %s, read done", in_file, out_file)
  data = msgpack.packb(good)
  data_compressed = zlib.compress(data)
  # zlib doesn't care about this
  data_compressed += "\x00" * (16 - len(data_compressed)%16)
  aes = AES.new(SWAG, AES.MODE_CBC, "\x00"*16)
  data_encrypted = aes.encrypt(data_compressed)
  with open(out_file, "wb") as outf:
    outf.write(data_encrypted)

  logging.debug("compressing %s -> %s, write done", in_file, out_file)

  return errors

def decompress_json_internal(data_encrypted):
  aes = AES.new(SWAG, AES.MODE_CBC, "\x00"*16)
  data_compressed = aes.decrypt(data_encrypted)
  data = zlib.decompress(data_compressed)
  msgs = msgpack.unpackb(data)
  good = []
  last_can_time = 0
  for ll in msgs:
    if ll[0][1] == 1:
      # back into hex
      ll[1][2] = ll[1][2].encode("hex")
      # derelativize CAN timestamps
      last_can_time += ll[1][1]
      ll[1] = [ll[1][0], last_can_time, ll[1][2]]
    good.append(ll)
  return good

def decompress_json(in_file, out_file):
  logging.debug("decompressing %s -> %s", in_file, out_file)
  f = open(in_file)
  data_encrypted = f.read()
  f.close()

  good = decompress_json_internal(data_encrypted)
  out = '\n'.join(map(lambda x: json.dumps(x), good)) + "\n"
  logging.debug("decompressing %s -> %s, writing", in_file, out_file)
  f = open(out_file, 'w')
  f.write(out)
  f.close()
  logging.debug("decompressing %s -> %s, write finished", in_file, out_file)

if __name__ == "__main__":
  for dat in sys.argv[1:]:
    print(dat)
    compress_json(dat, "/tmp/out"+ext)
    decompress_json("/tmp/out"+ext, "/tmp/test")
    os.system("diff "+dat+" /tmp/test")

