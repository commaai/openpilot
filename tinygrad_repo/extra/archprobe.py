# copying the kernels from https://github.com/microsoft/ArchProbe into Python
import numpy as np
import pickle
from tinygrad.runtime.ops_gpu import CLProgram, CLBuffer
from tinygrad import dtypes
from tqdm import trange, tqdm
from matplotlib import pyplot as plt

tests = {}
def register_test(fxn):
  tests[fxn.__name__] = fxn

def warp_size2(nthread):
  prg = """__kernel void warp_size2(
    __global float* src,
    __global int* dst,
    const int niter,
    const int prime_number
  ) {
    int drain = 0;
    for (int j = 0; j < niter; ++j) {
      drain += j / prime_number;
      barrier(0);
    }
    dst[get_local_id(0)] = drain;
  }"""
  src_buf = CLBuffer(1, dtypes.float32)
  dst_buf = CLBuffer(1, dtypes.int32)
  cl = CLProgram("warp_size2", prg, argdtypes=[None, None, np.int32, np.int32])
  return min([cl([nthread, 1024, 1], [nthread, 1, 1], src_buf, dst_buf, 10, 3, wait=True) for _ in range(5)])*1e9

@register_test
def test_warp_size():
  return [(nthread, warp_size2(nthread)) for nthread in trange(1,256)]

def reg_count(nthread, ngrp, nreg):
  reg_declr = ''.join([f"float reg_data{i} = (float)niter + {i};\n" for i in range(nreg)])
  reg_comp = ''.join([f"reg_data{i} *= {(i-1)%nreg};\n" for i in range(nreg)])
  reg_reduce = ''.join([f"out_buf[{i}] = reg_data{i};\n" for i in range(nreg)])
  prg = f"""__kernel void reg_count(
    __global float* out_buf,
    __private const int niter
  ) {{
    {reg_declr}
    int i = 0;
    for (; i < niter; ++i) {{
      {reg_comp}
    }}
    i = i >> 31;
    {reg_reduce}
  }}"""
  out_buf = CLBuffer(1, dtypes.float32)
  cl = CLProgram("reg_count", prg, argdtypes=[None, np.int32])
  return min([cl([nthread, ngrp, 1], [nthread, 1, 1], out_buf, 20, wait=True) for _ in range(10)])*1e9

@register_test
def test_reg_count(nthread=1, ngrp=1):
  base = reg_count(nthread, ngrp, 1)
  return [(nreg, (reg_count(nthread, ngrp, nreg)-base)/nreg) for nreg in trange(4, 513, 4)]

def buf_cache_hierarchy_pchase(ndata, stride=1, NCOMP=1, steps=65536):
  ndata //= NCOMP*4  # ptr size
  prg = f"""__kernel void buf_cache_hierarchy_pchase(
    __global int{str(NCOMP) if NCOMP > 1 else ''}* src,
    __global int* dst,
    const int niter
  ) {{
    int idx = 0;
    for (int i = 0; i < niter; ++i) {{
      idx = src[idx]{'.x' if NCOMP > 1 else ''};
    }}
    *dst = idx;
  }}"""
  idx_buf = np.zeros(ndata*NCOMP, dtype=np.int32)
  for i in range(ndata): idx_buf[i*NCOMP] = (i + stride) % ndata
  in_buf = CLBuffer.fromCPU(idx_buf)
  out_buf = CLBuffer(1, dtypes.int32)
  cl = CLProgram("buf_cache_hierarchy_pchase", prg, argdtypes=[None, None, np.int32])
  return min([cl([1, 1, 1], [1, 1, 1], in_buf, out_buf, steps, wait=True)/steps for _ in range(5)])*1e9

@register_test
def test_memory_latency():
  # requires cacheline < 16
  szs = [int(1.3**x) for x in range(20, 70)]
  return [(ndata, buf_cache_hierarchy_pchase(ndata, NCOMP=16, steps=128*1024)) for ndata in tqdm(szs)]

@register_test
def test_cacheline_size():
  # TODO: this buffer must be at least 2x the L1 cache for this test to work
  return [(stride, buf_cache_hierarchy_pchase(4*65536, stride, steps=65536)) for stride in trange(1,64)]

def cl_read(sz, niter=1):
  prg = f"""__kernel void copy(
    __global float4* src,
    __global float* dst) {{
      int gid = get_global_id(0);
      if (src[gid].x == 99+get_global_id(1)) *dst = 1;
  }}"""

  in_buf = CLBuffer(sz//4, dtypes.float32)
  out_buf = CLBuffer(1, dtypes.float32)
  cl = CLProgram("copy", prg)
  # NOTE: if nay of the niters form a local group, this is wrong
  return min([cl([sz//16, niter, 1], [1, 1, 1], in_buf, out_buf, wait=True) for _ in range(10)])*1e9

@register_test
def test_read_bandwidth():
  szs = list(range(128*1024, 20*1024*1024, 128*1024))
  NITER = 8
  base = cl_read(16, niter=NITER)
  return [(sz, (sz*NITER)/(cl_read(sz, niter=NITER)-base)) for sz in tqdm(szs)]


def gflops(niter=4, nroll=4, ngroups=4096):
  NCOMP = 8
  prg = f"""__kernel void gflops(
    __global float* out_buf
    ) {{
      float{NCOMP} x = (float{NCOMP})({",".join(f"get_local_id(0)+{i}" for i in range(NCOMP))});
      float{NCOMP} y = (float{NCOMP})({",".join(f"get_local_id(1)+{i}" for i in range(NCOMP))});

      for (int i = 0; i < {niter}; i++) {{
        {''.join(['x = mad(y, y, x); y = mad(x, x, y);'+chr(10)]*nroll)}
      }}

      out_buf[get_global_id(0) >> 31] = {'+'.join(f"y.s{'0123456789abcdef'[i]}" for i in range(NCOMP))};
  }}"""
  out_buf = CLBuffer(1, dtypes.float32)
  cl = CLProgram("gflops", prg, options="-cl-mad-enable -cl-fast-relaxed-math")
  FLOPS = NCOMP*2*2 * niter * nroll * ngroups * 32
  # NOTE: if nay of the niters form a local group, this is wrong
  return FLOPS/(min([cl([32, ngroups, 1], [32, 1, 1], out_buf, wait=True) for _ in range(10)])*1e9)

@register_test
def test_gflops():
  return [(niter, gflops(niter=niter, nroll=32)) for niter in trange(1, 32, 1)]

if __name__ == "__main__":
  cache = {}
  #cache = pickle.load(open("/tmp/cache.pkl", "rb"))
  #tests = {"test_cacheline_size": tests["test_cacheline_size"]}
  plt.figure(figsize=(16, 9))
  for i,(k,test) in enumerate(tests.items()):
    print(f"running {k}")
    plt.subplot(2, (len(tests)+1)//2, i+1)
    plt.title(k)
    if k == "test_memory_latency": plt.xscale('log')
    if k not in cache: cache[k] = test()
    plt.plot(*zip(*cache[k]))
  #pickle.dump(cache, open("/tmp/cache.pkl", "wb"))

  plt.tight_layout(pad=0.5)
  plt.savefig("/tmp/results.png")
  plt.show()
