#!/usr/bin/env python3
"""Benchmark comparing Python vs Rust RDNA3 emulators on synthetic and real tinygrad kernels."""
import ctypes, time, os, struct, cProfile, pstats, io
from pathlib import Path
from typing import Callable

# Set AMD=1 before importing tinygrad
os.environ["AMD"] = "1"

from extra.assembly.rdna3.emu import run_asm as python_run_asm, set_valid_mem_ranges, decode_program, step_wave, WaveState, WAVE_SIZE

REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.so"
if not REMU_PATH.exists():
  REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.dylib"

def get_rust_remu():
  """Load the Rust libremu shared library."""
  if not REMU_PATH.exists(): return None
  remu = ctypes.CDLL(str(REMU_PATH))
  remu.run_asm.restype = ctypes.c_int32
  remu.run_asm.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
  return remu

def count_instructions(kernel: bytes) -> int:
  """Count instructions in a kernel."""
  return len(decode_program(kernel))

def setup_buffers(buf_sizes: list[int], init_data: dict[int, bytes] | None = None):
  """Allocate buffers and return args pointer + valid ranges."""
  if init_data is None: init_data = {}
  buffers = []
  for i, size in enumerate(buf_sizes):
    padded = ((size + 15) // 16) * 16 + 16
    data = init_data.get(i, b'\x00' * padded)
    data_list = list(data) + [0] * (padded - len(data))
    buf = (ctypes.c_uint8 * padded)(*data_list[:padded])
    buffers.append(buf)
  args = (ctypes.c_uint64 * len(buffers))(*[ctypes.addressof(b) for b in buffers])
  args_ptr = ctypes.addressof(args)
  ranges = {(ctypes.addressof(b), len(b)) for b in buffers}
  ranges.add((args_ptr, ctypes.sizeof(args)))
  return buffers, args, args_ptr, ranges

def benchmark_emulator(name: str, run_fn, kernel: bytes, global_size, local_size, args_ptr, iterations: int = 5):
  """Benchmark an emulator and return average time."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  kernel_buf = (ctypes.c_char * len(kernel)).from_buffer_copy(kernel)
  lib_ptr = ctypes.addressof(kernel_buf)

  # Warmup
  run_fn(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)

  # Timed runs
  times = []
  for _ in range(iterations):
    start = time.perf_counter()
    result = run_fn(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)
    end = time.perf_counter()
    if result != 0:
      print(f"  {name} returned error: {result}")
      return None
    times.append(end - start)

  return sum(times) / len(times)

def create_synthetic_kernel(n_ops: int) -> bytes:
  """Create a synthetic kernel with n_ops vector operations."""
  instructions = []
  # VOP2 instructions: v_add_f32, v_mul_f32, v_max_f32, v_min_f32
  ops = [
    (0b0000011 << 25) | (1 << 17) | (0 << 9) | 256,  # v_add_f32 v0, v0, v1
    (0b0001000 << 25) | (1 << 17) | (0 << 9) | 256,  # v_mul_f32 v0, v0, v1
    (0b0010000 << 25) | (1 << 17) | (0 << 9) | 256,  # v_max_f32 v0, v0, v1
    (0b0001111 << 25) | (1 << 17) | (0 << 9) | 256,  # v_min_f32 v0, v0, v1
  ]
  for i in range(n_ops):
    instructions.append(ops[i % len(ops)])
  # S_ENDPGM
  instructions.append((0b101111111 << 23) | (48 << 16) | 0)
  return b''.join(struct.pack('<I', inst) for inst in instructions)

def get_tinygrad_kernel(op_name: str) -> tuple[bytes, tuple, tuple, list[int], dict[int, bytes]] | None:
  """Get a real tinygrad kernel by operation name. Returns (code, global_size, local_size, buf_sizes, buf_data)."""
  try:
    from tinygrad import Tensor
    from tinygrad.runtime.support.elf import elf_loader
    import numpy as np
    np.random.seed(42)

    ops = {
      "add": lambda: Tensor.empty(1024) + Tensor.empty(1024),
      "mul": lambda: Tensor.empty(1024) * Tensor.empty(1024),
      "matmul_small": lambda: Tensor.empty(16, 16) @ Tensor.empty(16, 16),
      "matmul_medium": lambda: Tensor.empty(64, 64) @ Tensor.empty(64, 64),
      "reduce_sum": lambda: Tensor.empty(4096).sum(),
      "reduce_max": lambda: Tensor.empty(4096).max(),
      "softmax": lambda: Tensor.empty(256).softmax(),
      "layernorm": lambda: Tensor.empty(32, 64).layernorm(),
      "conv2d": lambda: Tensor.empty(1, 4, 16, 16).conv2d(Tensor.empty(4, 4, 3, 3)),
      "gelu": lambda: Tensor.empty(1024).gelu(),
      "exp": lambda: Tensor.empty(1024).exp(),
      "sin": lambda: Tensor.empty(1024).sin(),
    }

    if op_name not in ops: return None
    out = ops[op_name]()
    sched = out.schedule()

    for ei in sched:
      lowered = ei.lower()
      if ei.ast.op.name == 'SINK' and lowered.prg and lowered.prg.p.lib:
        lib = bytes(lowered.prg.p.lib)
        _, sections, _ = elf_loader(lib)
        for sec in sections:
          if sec.name == '.text':
            buf_sizes = [b.nbytes for b in lowered.bufs]
            # Get initial data from numpy arrays if available
            buf_data = {}
            for i, buf in enumerate(lowered.bufs):
              if hasattr(buf, 'base') and buf.base is not None and hasattr(buf.base, '_buf'):
                try: buf_data[i] = bytes(buf.base._buf)
                except: pass
            return (bytes(sec.content), tuple(lowered.prg.p.global_size), tuple(lowered.prg.p.local_size), buf_sizes, buf_data)
    return None
  except Exception as e:
    print(f"  Error getting kernel: {e}")
    return None

def profile_python_emu(kernel: bytes, global_size, local_size, args_ptr, n_runs: int = 1):
  """Profile the Python emulator to find bottlenecks."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  kernel_buf = (ctypes.c_char * len(kernel)).from_buffer_copy(kernel)
  lib_ptr = ctypes.addressof(kernel_buf)

  pr = cProfile.Profile()
  pr.enable()
  for _ in range(n_runs):
    python_run_asm(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)
  pr.disable()

  s = io.StringIO()
  ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
  ps.print_stats(20)
  return s.getvalue()

def measure_step_rate(kernel: bytes, n_steps: int = 10000) -> float:
  """Measure raw step_wave() performance (steps per second)."""
  program = decode_program(kernel)
  if not program: return 0.0

  st = WaveState()
  st.exec_mask = 0xffffffff
  lds = bytearray(65536)
  n_lanes = 32

  # Reset PC for each measurement
  start = time.perf_counter()
  for _ in range(n_steps):
    st.pc = 0
    while st.pc in program:
      result = step_wave(program, st, lds, n_lanes)
      if result == -1: break
  elapsed = time.perf_counter() - start
  return n_steps / elapsed if elapsed > 0 else 0

# Test configurations
SYNTHETIC_TESTS = [
  ("synthetic_10ops", 10, (1, 1, 1), (32, 1, 1)),
  ("synthetic_100ops", 100, (1, 1, 1), (32, 1, 1)),
  ("synthetic_500ops", 500, (1, 1, 1), (32, 1, 1)),
  ("synthetic_100ops_4wg", 100, (4, 1, 1), (32, 1, 1)),
  ("synthetic_100ops_16wg", 100, (16, 1, 1), (32, 1, 1)),
]

TINYGRAD_TESTS = ["add", "mul", "reduce_sum", "softmax", "exp", "gelu", "matmul_small"]

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Benchmark RDNA3 emulators")
  parser.add_argument("--profile", action="store_true", help="Profile Python emulator")
  parser.add_argument("--synthetic-only", action="store_true", help="Only run synthetic tests")
  parser.add_argument("--tinygrad-only", action="store_true", help="Only run tinygrad tests")
  parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per benchmark")
  args = parser.parse_args()

  rust_remu = get_rust_remu()
  if rust_remu is None:
    print("Rust libremu not found. Build with: cargo build --release --manifest-path extra/remu/Cargo.toml")
    print("Running Python-only benchmarks...\n")

  print("=" * 90)
  print("RDNA3 Emulator Benchmark: Python vs Rust")
  print("=" * 90)

  results = []

  # Synthetic workloads
  if not args.tinygrad_only:
    print("\n[SYNTHETIC WORKLOADS]")
    print("-" * 90)

    for name, n_ops, global_size, local_size in SYNTHETIC_TESTS:
      kernel = create_synthetic_kernel(n_ops)
      n_insts = count_instructions(kernel)
      n_workgroups = global_size[0] * global_size[1] * global_size[2]
      n_threads = local_size[0] * local_size[1] * local_size[2]
      total_work = n_insts * n_workgroups * n_threads

      print(f"\n{name}: {n_insts} insts × {n_workgroups} WGs × {n_threads} threads = {total_work:,} ops")

      buf_sizes = [4096]
      buffers, args_arr, args_ptr, ranges = setup_buffers(buf_sizes)
      set_valid_mem_ranges(ranges)

      # Benchmark
      py_time = benchmark_emulator("Python", python_run_asm, kernel, global_size, local_size, args_ptr, args.iterations)
      rust_time = benchmark_emulator("Rust", rust_remu.run_asm, kernel, global_size, local_size, args_ptr, args.iterations) if rust_remu else None

      if py_time:
        py_rate = total_work / py_time / 1e6
        print(f"  Python: {py_time*1000:8.3f} ms  ({py_rate:7.2f} M ops/s)")
      if rust_time:
        rust_rate = total_work / rust_time / 1e6
        speedup = py_time / rust_time if py_time else 0
        print(f"  Rust:   {rust_time*1000:8.3f} ms  ({rust_rate:7.2f} M ops/s)  [{speedup:.1f}x faster]")

      results.append(("synthetic", name, n_insts, n_workgroups, py_time, rust_time))

  # Tinygrad kernels
  if not args.synthetic_only:
    print("\n[TINYGRAD KERNELS]")
    print("-" * 90)

    for op_name in TINYGRAD_TESTS:
      print(f"\n{op_name}:", end=" ", flush=True)
      kernel_info = get_tinygrad_kernel(op_name)
      if kernel_info is None:
        print("failed to compile")
        continue

      kernel, global_size, local_size, buf_sizes, buf_data = kernel_info
      n_insts = count_instructions(kernel)
      n_workgroups = global_size[0] * global_size[1] * global_size[2]
      n_threads = local_size[0] * local_size[1] * local_size[2]
      total_work = n_insts * n_workgroups * n_threads

      print(f"{n_insts} insts × {n_workgroups} WGs × {n_threads} threads = {total_work:,} ops")

      buffers, args_arr, args_ptr, ranges = setup_buffers(buf_sizes, buf_data)
      set_valid_mem_ranges(ranges)

      py_time = benchmark_emulator("Python", python_run_asm, kernel, global_size, local_size, args_ptr, args.iterations)
      rust_time = benchmark_emulator("Rust", rust_remu.run_asm, kernel, global_size, local_size, args_ptr, args.iterations) if rust_remu else None

      if py_time:
        py_rate = total_work / py_time / 1e6
        print(f"  Python: {py_time*1000:8.3f} ms  ({py_rate:7.2f} M ops/s)")
      if rust_time:
        rust_rate = total_work / rust_time / 1e6
        speedup = py_time / rust_time if py_time else 0
        print(f"  Rust:   {rust_time*1000:8.3f} ms  ({rust_rate:7.2f} M ops/s)  [{speedup:.1f}x faster]")

      results.append(("tinygrad", op_name, n_insts, n_workgroups, py_time, rust_time))

      # Optional profiling
      if args.profile and py_time:
        print("\n  [PROFILE - Top 10 functions]")
        profile_output = profile_python_emu(kernel, global_size, local_size, args_ptr)
        for line in profile_output.split('\n')[5:15]:
          if line.strip(): print(f"    {line}")

  # Summary table
  print("\n" + "=" * 90)
  print("SUMMARY")
  print("=" * 90)
  print(f"{'Type':<10} {'Name':<25} {'Insts':<8} {'WGs':<6} {'Python (ms)':<14} {'Rust (ms)':<14} {'Speedup':<10}")
  print("-" * 90)

  for test_type, name, n_insts, n_wgs, py_time, rust_time in results:
    py_ms = f"{py_time*1000:.3f}" if py_time else "error"
    if rust_time:
      rust_ms = f"{rust_time*1000:.3f}"
      speedup = f"{py_time/rust_time:.1f}x" if py_time else "N/A"
    else:
      rust_ms, speedup = "N/A", "N/A"
    print(f"{test_type:<10} {name:<25} {n_insts:<8} {n_wgs:<6} {py_ms:<14} {rust_ms:<14} {speedup:<10}")



if __name__ == "__main__":
  main()
