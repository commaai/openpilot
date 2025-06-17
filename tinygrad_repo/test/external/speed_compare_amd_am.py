from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, colorize_float, DEBUG
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from test.external.fuzz_linearizer import get_fuzz_rawbufs
from tinygrad.codegen.heuristic import hand_coded_optimizations
from tinygrad.engine.search import bufs_from_lin
from tinygrad.engine.realize import CompiledRunner
from tinygrad.tensor import _to_np_dtype
from tinygrad.runtime.ops_amd import AMDDevice
from contextlib import contextmanager
import numpy as np
import os, random, statistics

am_signal_pages, am_signal_pool, am_devices = [], [], []
amd_signal_pages, amd_signal_pool, amd_devices = [], [], []

def rebind_vfio(pcibus="0000:44:00.0"):
  print("rebind ", pcibus)
  os.system("sudo rmmod amdgpu")
  os.system("sudo modprobe vfio-pci")

  base = f"/sys/bus/pci/devices/{pcibus}"
  if os.path.exists(f"{base}/driver"):
    with open(f"{base}/driver/unbind", "w") as f: f.write(pcibus)
  with open(f"{base}/driver_override", "w") as f: f.write("vfio-pci")
  with open("/sys/bus/pci/drivers_probe", "w") as f: f.write(pcibus)

  os.system("sudo modprobe amdgpu")
  os.system("rocm-smi --setprofile compute")
  os.system("rocm-smi --setperflevel high")

@contextmanager
def run_amd():
  global amd_signal_pages, amd_signal_pool, amd_devices
  AMDDevice.driverless = False
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = amd_signal_pages, amd_signal_pool, amd_devices
  yield
  amd_signal_pages, amd_signal_pool, amd_devices = AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = [], [], []

@contextmanager
def run_am():
  global am_signal_pages, am_signal_pool, am_devices
  AMDDevice.driverless = True
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = am_signal_pages, am_signal_pool, am_devices
  yield
  am_signal_pages, am_signal_pool, am_devices = AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = [], [], []

if __name__ == "__main__":
  CHECK_CPU = getenv("CHECK_CPU", 0)
  SEED = getenv("SEED", 42)
  CNT = getenv("CNT", 7)
  random.seed(SEED)
  np.random.seed(SEED)

  # TODO: NUM=780 is super slow
  # NUM=1907 is broken on AMD and AM have some mismatches (0 vs 1)
  # kfd feels so bad when taking gpu out while it's running... Need hacks to rebind it before running.
  rebind_vfio(pcibus="0000:44:00.0")

  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)

  with run_am():
    amdev = Device["AMD:1"]

  with run_amd():
    amddev = Device["AMD"]

  if CHECK_CPU: cpudev = Device["CPU"]

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_amd, average_tm_am = 0, 0
  for num,ast in enumerate(ast_strs):
    with run_amd():
      amdlin = ast_str_to_lin(ast, opts=amddev.renderer)
      amdlin.apply_opts(hand_coded_optimizations(amdlin))
      has_bf16 = any(b.dtype == dtypes.bfloat16 for b in amdlin.membufs)

      amd_prg = CompiledRunner(amdlin.to_program())
      amdbufs = bufs_from_lin(amdlin)
      test_amdbufs = get_fuzz_rawbufs(amdlin) if not has_bf16 else amdbufs
      if not has_bf16: contents = [buf.as_buffer() for buf in test_amdbufs]

    with run_am():
      rdr = amdev.renderer
      rdr.device = "AMD:1"
      amlin = ast_str_to_lin(ast, opts=amdev.renderer)
      amlin.apply_opts(hand_coded_optimizations(amlin))
      am_prg = CompiledRunner(amlin.to_program())
      ambufs = bufs_from_lin(amlin)
      test_ambufs = get_fuzz_rawbufs(amlin) if not has_bf16 else ambufs
      if not has_bf16:
        for i,rawbuf in enumerate(test_ambufs): rawbuf.copyin(contents[i])

    if CHECK_CPU:
      cpu_rdr = cpudev.renderer
      cpu_rdr.device = "CPU"
      cpulin = ast_str_to_lin(ast, opts=cpu_rdr)
      cpulin.apply_opts(hand_coded_optimizations(cpulin))
      cpu_prg = CompiledRunner(cpulin.to_program())
      cpubufs = bufs_from_lin(cpulin)
      test_cpubufs = get_fuzz_rawbufs(cpulin) if not has_bf16 else ambufs
      if not has_bf16:
        for i,rawbuf in enumerate(test_cpubufs): rawbuf.copyin(contents[i])

    # warmup
    tm_amd, tm_am, failed = [], [], False
    with run_amd():
      try:
        amd_prg(test_amdbufs, {}, wait=True)
        for i in range(CNT): tm_amd.append(amd_prg(amdbufs, {}, wait=True))
      except RuntimeError:
        print("AMD FAILED")
        tm_amd = [1e9]
        failed = True

    with run_am():
      try:
        am_prg(test_ambufs, {}, wait=True)
        for i in range(CNT): tm_am.append(am_prg(ambufs, {}, wait=True))
      except RuntimeError:
        print("AM FAILED")
        tm_am = [1e9]
        failed = True

    if CHECK_CPU:
      cpu_prg(test_cpubufs, {}, wait=True)
      for i in range(1): cpu_prg(cpubufs, {}, wait=True)

    if not failed and not has_bf16:
      with run_amd():
        curesult = np.frombuffer(test_amdbufs[0].as_buffer(), _to_np_dtype(test_amdbufs[0].dtype))

      with run_am():
        amresult = np.frombuffer(test_ambufs[0].as_buffer(), _to_np_dtype(test_ambufs[0].dtype))

      if CHECK_CPU:
        cpuresult = np.frombuffer(test_cpubufs[0].as_buffer(), _to_np_dtype(test_cpubufs[0].dtype))
        np.testing.assert_allclose(amresult, cpuresult, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(curesult, cpuresult, rtol=1e-2, atol=1e-2)

      try:
        np.testing.assert_allclose(curesult, amresult, rtol=1e-2, atol=1e-2)
      except AssertionError as e:
        print("AM and AMD results do not match")
        print(e)

    bam = statistics.median(tm_am)
    bamd = statistics.median(tm_amd)
    average_tm_amd += bamd
    average_tm_am += bam
    ratio = bam/bamd
    print(f"{average_tm_am/average_tm_amd:5.2f}x -- {num:4d} {colorize_float(ratio)} {bam*1e6:7.2f} vs {bamd*1e6:7.2f} us", amlin.name)
    if DEBUG > 3 and ratio > 1.04: print(f"AM slower {ratio}", amlin.ast, amlin.applied_opts)
