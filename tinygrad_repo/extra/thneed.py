# this can be constructed from a cl_cache or loaded from a thneed file 
import os
import time
import struct
import json
import traceback
import numpy as np
from tinygrad.llops.ops_gpu import CL, CLProgram
from tinygrad.helpers import prod
from collections import defaultdict
import pyopencl as cl

DEBUGCL = int(os.getenv("DEBUGCL", 0))
FLOAT16 = int(os.getenv("FLOAT16", 0))

class Thneed:
  def __init__(self, cl_cache=[], inputs={}):
    self.cl_cache, self.inputs = cl_cache[:], inputs
    self.gobj = 0

    # build graph
    # NOTE: if CLCACHE=1, this is wrong!
    nodes = defaultdict(lambda: {'in_edges': [], 'out_edges': []})
    for _, args in self.cl_cache:
      # output is always the first parameter
      for a in args[3:]:
        nodes[a]['out_edges'].append(args[2])
        nodes[args[2]]['in_edges'].append(a)
    
    # get buffers to save
    self.buffers_to_save = set()
    self.outputs = []
    for n in nodes.keys():
      if len(nodes[n]['in_edges']) == 0:
        self.buffers_to_save.add(n)
      if len(nodes[n]['out_edges']) == 0:
        self.outputs.append(n)
  
    fake_inputs = []
    for k,n in self.inputs.items():
      if n in self.buffers_to_save:
        self.buffers_to_save.remove(n)
      else:
        print(f"WARNING: {k} was not a used input, removing it")
        fake_inputs.append(k)
    for k in fake_inputs:
      del self.inputs[k]

  def load(self, input_fn):
    float32 = not FLOAT16

    mf = cl.mem_flags
    image_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT if float32 else cl.channel_type.HALF_FLOAT)
    image_fmt_32 = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    with open(input_fn, "rb") as f:
      json_len = struct.unpack("I", f.read(4))[0]
      jdat = json.loads(f.read(json_len).decode('latin_1'))
      weights = f.read()

    # load in the buffers
    bufs = {'\x00\x00\x00\x00\x00\x00\x00\x00': None}
    bufs_loaded = {}
    ptr = 0
    for o in jdat['objects']:
      #print(o)
      if o['needs_load']:
        nptr = ptr + o['size']
        o['data'] = weights[ptr:nptr]
        ptr = nptr

      if o['arg_type'] == "image2d_t" or o['arg_type'] == "image1d_t":
        tfmt = image_fmt_32 if 'float32' in o and o['float32'] else image_fmt
        if o['arg_type'] == "image2d_t":
          if 'buffer_id' in o and o['height'] == 1 and not bufs_loaded[o['buffer_id']]:
            # hack: use a image1d since we can back that with a buffer
            buf = cl.Image(CL().cl_ctx, mf.READ_WRITE, tfmt, shape=(o['width'],), buffer=bufs[o['buffer_id']])
          else:
            # buffer isn't supported in image2d, copy buffer into image
            if 'buffer_id' in o and bufs_loaded[o['buffer_id']]:
              arr = np.zeros(bufs[o['buffer_id']].size // 2, dtype=np.float16)
              cl.enqueue_copy(q, arr, bufs[o['buffer_id']])
              buf = cl.Image(CL().cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, tfmt,
                shape=(o['width'], o['height']), pitches=(o['row_pitch'],), hostbuf=arr)
            elif o['needs_load']:
              buf = cl.Image(CL().cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, tfmt,
                shape=(o['width'], o['height']), pitches=(o['row_pitch'],), hostbuf=o['data'])
            else:
              buf = cl.Image(CL().cl_ctx, mf.READ_WRITE, tfmt, shape=(o['width'], o['height']))
        if o['arg_type'] == "image1d_t":
          assert not o['needs_load']
          assert not bufs_loaded[o['buffer_id']]
          buf = cl.Image(CL().cl_ctx, mf.READ_WRITE, tfmt, shape=(o['width'],), buffer=bufs[o['buffer_id']])
      else:
        if 'data' in o:
          buf = cl.Buffer(CL().cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=o['data'])
        else:
          # zero out buffers
          buf = cl.Buffer(CL().cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b'\x00'*o['size'])
        
      bufs[o['id']] = buf
      bufs_loaded[o['id']] = 'data' in o
      # if it's loaded, it's saved
      if 'data' in o:
        self.buffers_to_save.add(buf)

    # load in the programs (this isn't used)
    prgs = {}
    for k,v in jdat['programs'].items():
      print("building", k)
      try:
        prgs[k] = CLProgram(k, v, rename=False)
      except Exception:
        print("FAILED", k)
        traceback.print_exc()
        exit(0)
    
    # load binaries
    for o in jdat['binaries']:
      nptr = ptr + o['length']
      prgs[o['name']] = CLProgram(o['name'], weights[ptr:nptr], rename=False, binary=True)
      ptr = nptr
  
    # populate the cl_cache
    for i,k in enumerate(jdat['kernels']):
      kernel = prgs[k['name']]
      aaa = []
      for j,(a,sz) in enumerate(zip(k['args'], k['args_size'])):
        if len(a) == 0:
          aa = cl.LocalMemory(sz)
        elif len(a) == 4:
          a = a.encode('latin_1')
          aa = np.uint32(struct.unpack("I", a)[0])
        elif len(a) == 2:
          a = a.encode('latin_1')
          aa = np.uint16(struct.unpack("H", a)[0])
        elif len(a) == 8:
          #print(i,j,struct.unpack("Q", a.encode('latin_1'))[0])
          aa = bufs[a]
        aaa.append(aa)
      self.cl_cache.append((kernel, [k['global_work_size'], k['local_work_size'], *aaa]))

    # load inputs
    for k in jdat['inputs']:
      self.inputs[k['name']] = bufs[k['buffer_id']]

    # load outputs
    for k in jdat['outputs']:
      self.outputs.append(bufs[k['buffer_id']])


  def save(self, output_fn):
    # this is the struct that will be saved
    jdat = {"binaries": [], "programs": {}, "kernels": [], "objects": []}

    # build the pieces of this struct
    weights = []
    binaries = []
    saved_objs = set()
    saved_binaries = set()
    for prg, args in self.cl_cache:
      # get binaries for saving
      if prg.name not in saved_binaries:
        binary = prg.clprogram.get_info(cl.program_info.BINARIES)
        assert len(binary) == 1
        jdat['binaries'].append({"name":prg.name, "length":len(binary[0])})
        binaries.append(binary[0])
        saved_binaries.add(prg.name)
      
      # get the args from the kernel, some need the data saved
      targs, args_size = [], []
      argdtypes = prg.argdtypes if prg.argdtypes is not None else [None]*(len(args)-2)
      for a,d in zip(args[2:], argdtypes):
        if d == np.int16:
          targs.append(struct.pack("H", a).decode("latin_1"))
          args_size.append(2)
        elif d == np.int32:
          targs.append(struct.pack("I", a).decode("latin_1"))
          args_size.append(4)
        elif isinstance(a, cl.LocalMemory):
          targs.append("")
          args_size.append(a.size)
        elif d is None:
          if getattr(a, "global_id", None) is None:
            setattr(a, "global_id", self.gobj)
            self.gobj += 1
          ptr = struct.pack("Q", a.global_id).decode("latin_1")
          if ptr not in saved_objs:
            if isinstance(a, cl.Buffer):
              needs_load = a in self.buffers_to_save
              jdat['objects'].append({
                "id": ptr, "arg_type": "float*", "needs_load": needs_load, "size": a.size,
              })
              if needs_load:
                data = np.empty(a.size//4, dtype=np.float32)
                cl.enqueue_copy(CL().cl_queue, data, a, is_blocking=True)
                weights.append(data.tobytes())
            elif isinstance(a, cl.Image):
              needs_load = a in self.buffers_to_save
              row_pitch = (a.shape[0]*4*(2 if FLOAT16 else 4) + 63)//64 * 64
              size = row_pitch * a.shape[1]
              # this is *2 if float16 and *4 if float32
              buf = cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, size=size * (2 if FLOAT16 else 1))

              # zero out the buffer
              cl.enqueue_copy(CL().cl_queue, buf, b'\x00'*buf.size, is_blocking=True)

              CLProgram("from_image_strided", """
                __kernel void from_image_strided(read_only image2d_t in, __global float4 *out, int row_pitch) {
                  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
                  int2 l;
                  l.y = get_global_id(1);
                  l.x = get_global_id(0);
                  out[l.y*row_pitch + l.x] = read_imagef(in, smp, l);
                }
              """, argdtypes=(None, None, np.int32))(a.shape, None, a, buf, row_pitch//(4*(2 if FLOAT16 else 4)))

              # multiple of 32 isn't enough
              jdat['objects'].append({
                "id": ptr, "needs_load": needs_load, "size": size, "arg_type": "image2d_t",
                "width": a.shape[0], "height": a.shape[1], "row_pitch": row_pitch, "float32": not FLOAT16,
              })

              if needs_load:
                data = np.empty(size//(2 if FLOAT16 else 4), dtype=np.float32)
                cl.enqueue_copy(CL().cl_queue, data, buf, is_blocking=True)
                if FLOAT16: data = data.astype(np.float16)
                weights.append(data.tobytes())
            else:
              raise Exception("unknown object", a)
            #print(jdat['objects'][-1])
            saved_objs.add(ptr)
          targs.append(ptr)
          args_size.append(8)
        else:
          raise Exception("idk this type")

      # save the kernel itself
      jdat['kernels'].append({
        "name": prg.name,
        "work_dim": len(args[0]),
        "global_work_size": args[0],
        # TODO: C++ thneed requires a local_work_size, so we fill it with ones
        "local_work_size": [1 for _ in args[0]] if args[1] is None else args[1],
        "num_args": len(args)-2,
        "args": targs,
        "args_size": args_size 
      })

    jdat['outputs'] = [{
      "buffer_id": struct.pack("Q", x.global_id).decode("latin_1"),
      "size": x.size,
    } for x in self.outputs]

    jdat['inputs'] = [{
      "buffer_id": struct.pack("Q", v.global_id).decode("latin_1"),
      "size": v.size,
      "name": k
    } for k,v in self.inputs.items()][::-1]

    print(f"saving thneed to {output_fn}")
    with open(output_fn, "wb") as f:
      j = json.dumps(jdat, ensure_ascii=False).encode('latin_1')
      f.write(struct.pack("I", len(j)))
      f.write(j)
      f.write(b''.join(weights))
      f.write(b''.join(binaries))

  def run(self):
    events = []
    st = time.monotonic()
    for prg, args in self.cl_cache:
      events.append(prg.clprg(CL().cl_queue, *args))
    mt = time.monotonic()
    CL().cl_queue.finish()
    et = time.monotonic() - st
    print(f"submit in {(mt-st)*1000.0:.2f} ms, total runtime is {et*1000.0:.2f} ms")

    if DEBUGCL >= 1:
      scale_total_runtime = sum([(e.profile.end - e.profile.start) for e in events])/1e9
      # TODO: Mac OS has a scaling issue, this hack fixes it
      scale = 1 if (et/scale_total_runtime) < 10 else (et/scale_total_runtime)
      total_runtime = 0
      for i, ((prg, args), e) in enumerate(zip(self.cl_cache, events)):
        runtime = (e.profile.end - e.profile.start)*scale
        print(f"{i:3d} time {total_runtime/1e6:5.2f} ms running {prg.name:20s} with {str(args[0]):15s} {str(args[1]):15s} count {len(args)-2:2d} runtime {runtime/1e3:7.2f} us {(prg.op_estimate)/runtime:9.2f} GFLOPS {prg.options} -> {args[2].shape if hasattr(args[2], 'shape') else args[2].size}")
        if (DEBUGCL >= 2 and int(os.getenv("PRINT_KERNEL", "-1")) == i) or DEBUGCL >= 3:
          print(prg.prg)
        total_runtime += runtime
      print(f"total runtime: {total_runtime/1e6:.2f} ms   wall time: {et*1000.0:.2f} ms")
    return et

  def optimize_local_workgroup(self):
    MAX_WORKGROUP = CL.cl_ctx.devices[0].max_work_group_size
    local_cl_cache = []
    for prg, args in self.cl_cache:
      potential_locals = [tuple(args[1])] if args[1] is not None else []
      runtimes = []
      args = list(args)

      if args[1] is None and len(args[0]) == 1:
        for l1 in [args[0][0], 1, 4, 16, MAX_WORKGROUP//4, MAX_WORKGROUP]:
          potential_locals.append((l1,))

      if args[1] is None and len(args[0]) == 2:
        for l2 in [1, 4, 16, MAX_WORKGROUP//4, MAX_WORKGROUP]:
          potential_locals.append((min(MAX_WORKGROUP, args[0][0]), l2))

      if args[1] is None and len(args[0]) == 3:
        for l2 in [16,args[0][1],MAX_WORKGROUP]:
          for l3 in [4,16,args[0][2],MAX_WORKGROUP]:
            for l1 in [max(1, MAX_WORKGROUP//(l2*l3)), args[0][0], 4, 16, MAX_WORKGROUP]:
              if l1 > args[0][0] or l2 > args[0][1] or l3 > args[0][2]: continue
              potential_locals.append((l1, l2, l3))

      # TODO: why does this introduce a bug?
      #if args[1] is not None and len(args[0]) == 3:
      #  for l2 in [args[1][1], args[0][1]]:  # might be required, whole thing is always safe
      #    for l3 in [4,16,args[0][2],MAX_WORKGROUP,args[1][2]]:
      #      potential_locals.append((args[1][0], l2, l3))

      for local_args in potential_locals:
        if prod(local_args) > MAX_WORKGROUP: continue
        args[1] = local_args
        # 3 runs just in case
        for i in range(3):
          try:
            e = prg.clprg(CL().cl_queue, *args)
          except (cl.LogicError, cl.RuntimeError):
            # INVALID_WORK_GROUP_SIZE
            continue
          CL().cl_queue.finish()
          runtime = e.profile.end - e.profile.start
          #print(runtime, args[0], args[1])
          runtimes.append((runtime, local_args))

        if len(runtimes) > 0:
          args[1] = sorted(runtimes)[0][1]
        else:
          args[1] = None
          print("couldn't optimize", args[0])

      local_cl_cache.append((prg, args))
    self.cl_cache = local_cl_cache
