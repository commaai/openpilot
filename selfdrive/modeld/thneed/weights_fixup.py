#!/usr/bin/env python3
import os, struct, json
from common.basedir import BASEDIR
import numpy as np

# load onnx weights
def get_onnx_weights():
  import onnx
  from onnx import helper, numpy_helper

  model = onnx.load(os.path.join(BASEDIR, "models/supercombo.onnx"))
  graph = model.graph
  init = {x.name:x for x in graph.initializer}

  onnx_layers = []
  for node in graph.node:
    #print(node.name)
    vals = []
    for inp in node.input:
      if inp in init:
        vals.append(numpy_helper.to_array(init[inp]))
    if len(vals) > 0:
      onnx_layers.append((node.name, vals))
  return onnx_layers

onnx_layers = get_onnx_weights()

# load thneed file

with open(os.path.join(BASEDIR, "models/supercombo.thneed"), "rb") as f:
  json_len = struct.unpack("I", f.read(4))[0]
  jdat = json.loads(f.read(json_len).decode('latin_1'))
  weights = f.read()

ptr = 0
bufs = {}
for o in jdat['objects']:
  if o['needs_load']:
    nptr = ptr + o['size']
    o['data'] = weights[ptr:nptr]
    ptr = nptr
  bufs[o['id']] = o

thneed_layers = []
for k in jdat['kernels']:
  #print(k['name'])
  vals = []
  for a in k['args']:
    if a in bufs:
      o = bufs[a]
      if o['needs_load'] or ('buffer_id' in o and bufs[o['buffer_id']]['needs_load']):
        #print("  ", o['arg_type'])
        vals.append(o)
  if len(vals) > 0:
    thneed_layers.append((k['name'], vals))

assert len(thneed_layers) == len(onnx_layers)

# fix up weights

for tl, ol in zip(thneed_layers, onnx_layers):
  print(tl[0], ol[0])
  assert len(tl[1]) == len(ol[1])
  for o, onnx_weight in zip(tl[1], ol[1]):
    if o['arg_type'] == "image2d_t":
      obuf = bufs[o['buffer_id']]
      saved_weights = np.frombuffer(obuf['data'], dtype=np.float16).reshape(o['height'], o['row_pitch']//2)

      # convolution
      if tl[0] in ["convolution_horizontal_reduced_reads", "convolution_horizontal_reduced_reads_1x1"]:
        oc,ic,ch,cw = onnx_weight.shape
        weights = np.transpose(onnx_weight.reshape(oc//4,4,ic//4,4,ch,cw), (0,4,2,5,1,3)).reshape(o['height'], o['width']*4)
        new_weights = np.zeros((o['height'], o['row_pitch']//2), dtype=np.float32)
        new_weights[:, :o['width']*4] = weights

        err = np.mean((saved_weights.astype(np.float32) - new_weights)**2)
        fixed_err = np.mean((new_weights.astype(np.float16).astype(np.float32) - new_weights)**2)

        print(o['size'], onnx_weight.shape, o['row_pitch'], o['width'], o['height'], "err %.2f x better" % (err/fixed_err))

        obuf['data'] = new_weights.astype(np.float16).tobytes()

# save thneed file

new_weights = []
for o in jdat['objects']:
  if 'data' in o:
    new_weights.append(o['data'])
    del o['data']
new_weights = b''.join(new_weights)

with open(os.path.join(BASEDIR, "models/supercombo_fixed.thneed"), "wb") as f:
  j = json.dumps(jdat, ensure_ascii=False).encode('latin_1')
  f.write(struct.pack("I", len(j)))
  f.write(j)
  f.write(new_weights)

exit(0)



