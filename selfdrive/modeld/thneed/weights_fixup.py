#!/usr/bin/env python3
import os
import struct
import zipfile
import numpy as np
from tqdm import tqdm

from common.basedir import BASEDIR
from selfdrive.modeld.thneed.lib import load_thneed, save_thneed

# this is junk code, but it doesn't have deps
def load_dlc_weights(fn):
  archive = zipfile.ZipFile(fn, 'r')
  dlc_params = archive.read("model.params")

  def extract(rdat):
    idx = rdat.find(b"\x00\x00\x00\x09\x04\x00\x00\x00")
    rdat = rdat[idx+8:]
    ll = struct.unpack("I", rdat[0:4])[0]
    buf = np.frombuffer(rdat[4:4+ll*4], dtype=np.float32)
    rdat = rdat[4+ll*4:]
    dims = struct.unpack("I", rdat[0:4])[0]
    buf = buf.reshape(struct.unpack("I"*dims, rdat[4:4+dims*4]))
    if len(buf.shape) == 4:
      buf = np.transpose(buf, (3,2,0,1))
    return buf

  def parse(tdat):
    ll = struct.unpack("I", tdat[0:4])[0] + 4
    return (None, [extract(tdat[0:]), extract(tdat[ll:])])

  ptr = 0x20
  def r4():
    nonlocal ptr
    ret = struct.unpack("I", dlc_params[ptr:ptr+4])[0]
    ptr += 4
    return ret
  ranges = []
  cnt = r4()
  for _ in range(cnt):
    o = r4() + ptr
    # the header is 0xC
    plen, is_4, is_2 = struct.unpack("III", dlc_params[o:o+0xC])
    assert is_4 == 4 and is_2 == 2
    ranges.append((o+0xC, o+plen+0xC))
  ranges = sorted(ranges, reverse=True)

  return [parse(dlc_params[s:e]) for s,e in ranges]

# this won't run on device without onnx
def load_onnx_weights(fn):
  import onnx
  from onnx import numpy_helper

  model = onnx.load(fn)
  graph = model.graph  # pylint: disable=maybe-no-member
  init = {x.name:x for x in graph.initializer}

  onnx_layers = []
  for node in graph.node:
    #print(node.name, node.op_type, node.input, node.output)
    vals = []
    for inp in node.input:
      if inp in init:
        vals.append(numpy_helper.to_array(init[inp]))
    if len(vals) > 0:
      onnx_layers.append((node.name, vals))
  return onnx_layers

def weights_fixup(target, source_thneed, dlc):
  #onnx_layers = load_onnx_weights(os.path.join(BASEDIR, "models/supercombo.onnx"))
  onnx_layers = load_dlc_weights(dlc)
  jdat = load_thneed(source_thneed)

  bufs = {}
  for o in jdat['objects']:
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
  for tl, ol in tqdm(zip(thneed_layers, onnx_layers), total=len(thneed_layers)):
    #print(tl[0], ol[0])
    assert len(tl[1]) == len(ol[1])
    for o, onnx_weight in zip(tl[1], ol[1]):
      if o['arg_type'] == "image2d_t":
        obuf = bufs[o['buffer_id']]
        saved_weights = np.frombuffer(obuf['data'], dtype=np.float16).reshape(o['height'], o['row_pitch']//2)

        if len(onnx_weight.shape) == 4:
          # convolution
          oc,ic,ch,cw = onnx_weight.shape

          if 'depthwise' in tl[0]:
            assert ic == 1
            weights = np.transpose(onnx_weight.reshape(oc//4,4,ch,cw), (0,2,3,1)).reshape(o['height'], o['width']*4)
          else:
            weights = np.transpose(onnx_weight.reshape(oc//4,4,ic//4,4,ch,cw), (0,4,2,5,1,3)).reshape(o['height'], o['width']*4)
        else:
          # fc_Wtx
          weights = onnx_weight

        new_weights = np.zeros((o['height'], o['row_pitch']//2), dtype=np.float32)
        new_weights[:, :weights.shape[1]] = weights

        # weights shouldn't be too far off
        err = np.mean((saved_weights.astype(np.float32) - new_weights)**2)
        assert err < 1e-3
        rerr = np.mean(np.abs((saved_weights.astype(np.float32) - new_weights)/(new_weights+1e-12)))
        assert rerr < 0.5

        # fix should improve things
        fixed_err = np.mean((new_weights.astype(np.float16).astype(np.float32) - new_weights)**2)
        assert (err/fixed_err) >= 1

        #print("   ", o['size'], onnx_weight.shape, o['row_pitch'], o['width'], o['height'], "err %.2fx better" % (err/fixed_err))

        obuf['data'] = new_weights.astype(np.float16).tobytes()

      elif o['arg_type'] == "float*":
        # unconverted floats are correct
        new_weights = np.zeros(o['size']//4, dtype=np.float32)
        new_weights[:onnx_weight.shape[0]] = onnx_weight
        assert new_weights.tobytes() == o['data']
        #print("   ", o['size'], onnx_weight.shape)

  save_thneed(jdat, target)

if __name__ == "__main__":
  weights_fixup(os.path.join(BASEDIR, "models/supercombo_fixed.thneed"),
                os.path.join(BASEDIR, "models/supercombo.thneed"),
                os.path.join(BASEDIR, "models/supercombo.dlc"))
