import struct
import json

def load_thneed(fn):
  with open(fn, "rb") as f:
    json_len = struct.unpack("I", f.read(4))[0]
    jdat = json.loads(f.read(json_len).decode('latin_1'))
    weights = f.read()
  ptr = 0
  for o in jdat['objects']:
    if o['needs_load']:
      nptr = ptr + o['size']
      o['data'] = weights[ptr:nptr]
      ptr = nptr
  for o in jdat['binaries']:
    nptr = ptr + o['length']
    o['data'] = weights[ptr:nptr]
    ptr = nptr
  return jdat

def save_thneed(jdat, fn):
  new_weights = []
  for o in jdat['objects'] + jdat['binaries']:
    if 'data' in o:
      new_weights.append(o['data'])
      del o['data']
  new_weights_bytes = b''.join(new_weights)
  with open(fn, "wb") as f:
    j = json.dumps(jdat, ensure_ascii=False).encode('latin_1')
    f.write(struct.pack("I", len(j)))
    f.write(j)
    f.write(new_weights_bytes)
