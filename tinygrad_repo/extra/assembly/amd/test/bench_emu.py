#!/usr/bin/env python3
"""Benchmark comparing Python vs Rust RDNA3 emulators on real tinygrad kernels."""
import ctypes, time, os
from pathlib import Path

# Set AMD=1 before importing tinygrad
os.environ["AMD"] = "1"

from extra.assembly.amd.emu2 import run_asm as python_run_asm, decode_program, _get_inst_sink, _get_inst_prg
from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP, SOPPOp

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

def benchmark_emulator(name: str, run_fn, kernel: bytes, global_size, local_size, args_ptr, rsrc2: int, iterations: int = 5):
  """Benchmark an emulator and return average time."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  kernel_buf = (ctypes.c_char * len(kernel)).from_buffer_copy(kernel)
  lib_ptr = ctypes.addressof(kernel_buf)

  # Warmup
  run_fn(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr, rsrc2)

  # Timed runs
  times = []
  for _ in range(iterations):
    start = time.perf_counter()
    result = run_fn(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr, rsrc2)
    end = time.perf_counter()
    if result != 0:
      print(f"  {name} returned error: {result}")
      return None
    times.append(end - start)

  return sum(times) / len(times)

def profile_instructions(kernel: bytes):
  """Profile individual instructions and return sorted by render time."""
  from extra.assembly.amd.emu2 import _get_inst_prg, _get_inst_sink, _canonical_prg_cache
  from tinygrad.codegen import get_program
  from extra.assembly.amd.emu2 import _emu_renderer
  from tinygrad.helpers import Context

  # Clear caches to measure fresh
  _get_inst_sink.cache_clear()
  _get_inst_prg.cache_clear()
  _canonical_prg_cache.clear()
  decode_program.cache_clear()

  # Collect instruction bytes and names
  inst_data = []
  i = 0
  while i < len(kernel):
    inst = decode_inst(kernel[i:])
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break
    inst_bytes = bytes(kernel[i:i + inst.size() + 4])
    try:
      inst_str = repr(inst)
    except Exception:
      inst_str = f"<{type(inst).__name__}>"
    inst_data.append((inst_bytes, inst_str, type(inst).__name__))
    i += inst.size()

  # Profile each instruction
  from extra.assembly.amd.emu2 import _match_canonical
  results = []
  for inst_bytes, inst_str, inst_type in inst_data:
    # Check canonical cache BEFORE building sink (matches real behavior)
    inst_size = decode_inst(inst_bytes).size()
    inst_int = int.from_bytes(inst_bytes[:inst_size], 'little')
    is_cache_hit = _match_canonical(inst_int, inst_size) is not None

    if is_cache_hit:
      # Skip build and render entirely for cache hits
      build_time, render_time, uop_count = 0, 0, 0
    else:
      # Build sink
      build_start = time.perf_counter()
      sink, ctx = _get_inst_sink(inst_bytes)
      build_time = time.perf_counter() - build_start

      # Count UOps in sink
      uop_count = len(sink.toposort())

      # Render
      render_start = time.perf_counter()
      with Context(NOOPT=1, IGNORE_OOB=1, TUPLE_ORDER=0):
        prg = get_program(sink, _emu_renderer)
      render_time = time.perf_counter() - render_start

      # Update canonical cache
      base, mask, size = ctx.canonical_mask(inst_bytes)
      _canonical_prg_cache.append((base, mask, size, prg))

    results.append({
      'inst_str': inst_str + (' [HIT]' if is_cache_hit else ''),
      'inst_type': inst_type,
      'uop_count': uop_count,
      'build_ms': build_time * 1000,
      'render_ms': render_time * 1000,
    })

  # Sort by render time descending
  return sorted(results, key=lambda x: x['render_ms'], reverse=True)

def benchmark_python_split(kernel: bytes, global_size, local_size, args_ptr, rsrc2: int, iterations: int = 5):
  """Benchmark Python emulator with build/render/compile/execution times separated."""
  from extra.assembly.amd.emu2 import _emu_renderer, _emu_compiler, _elf_symbol_offsets
  from extra.assembly.amd.emu2 import _get_inst_prg, _get_inst_sink, _canonical_prg_cache
  from tinygrad.codegen import get_program
  from tinygrad.helpers import Context
  from tinygrad.runtime.support.elf import jit_loader

  # Clear caches to measure fresh
  _get_inst_sink.cache_clear()
  _get_inst_prg.cache_clear()
  _canonical_prg_cache.clear()
  decode_program.cache_clear()

  # Collect instruction bytes
  inst_bytes_list = []
  i = 0
  while i < len(kernel):
    inst = decode_inst(kernel[i:])
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break
    inst_bytes_list.append(bytes(kernel[i:i + inst.size() + 4]))
    i += inst.size()

  # Measure build time (UOp sink generation, cached)
  build_start = time.perf_counter()
  for inst_bytes in inst_bytes_list:
    _get_inst_sink(inst_bytes)
  build_time = time.perf_counter() - build_start

  # Measure render time (uses cached sinks, handles canonical dedup)
  render_start = time.perf_counter()
  cache_before = len(_canonical_prg_cache)
  prgs = [_get_inst_prg(inst_bytes) for inst_bytes in inst_bytes_list]
  render_count = len(_canonical_prg_cache) - cache_before  # number of unique renders
  render_time = time.perf_counter() - render_start

  # Measure compile time (clang/llvm compile C to native)
  compile_start = time.perf_counter()
  # Deduplicate by function name (same as decode_program does)
  seen = set()
  unique_srcs = []
  for prg in prgs:
    if prg.function_name not in seen:
      seen.add(prg.function_name)
      unique_srcs.append(prg.src)
  combined_src = "\n".join(unique_srcs)
  obj = _emu_compiler.compile_to_obj(combined_src)
  _elf_symbol_offsets(obj)
  jit_loader(obj)
  compile_time = time.perf_counter() - compile_start

  # Execution time (need to populate cache first)
  decode_program(kernel)
  exec_time = benchmark_emulator("Python", python_run_asm, kernel, global_size, local_size, args_ptr, rsrc2, iterations)
  return build_time, render_time, render_count, compile_time, exec_time

def get_tinygrad_kernel(op_name: str) -> tuple[bytes, tuple, tuple, list[int], dict[int, bytes], int] | None:
  """Get a real tinygrad kernel by operation name. Returns (code, global_size, local_size, buf_sizes, buf_data, rsrc2)."""
  try:
    from tinygrad import Tensor
    from tinygrad.runtime.support.elf import elf_loader
    from tinygrad.runtime.autogen import hsa
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
        image = memoryview(bytearray(lib))
        _, sections, _ = elf_loader(lib)
        rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
        for sec in sections:
          if sec.name == '.text':
            buf_sizes = [b.nbytes for b in lowered.bufs]
            # Get initial data from numpy arrays if available
            buf_data = {}
            for i, buf in enumerate(lowered.bufs):
              if hasattr(buf, 'base') and buf.base is not None and hasattr(buf.base, '_buf'):
                try: buf_data[i] = bytes(buf.base._buf)
                except: pass
            # Extract rsrc2 from ELF (same as ops_amd.py)
            group_segment_size = image[rodata_entry:rodata_entry+4].cast("I")[0]
            lds_size = ((group_segment_size + 511) // 512) & 0x1FF
            code = hsa.amd_kernel_code_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+256]) + b'\x00'*256)
            rsrc2 = code.compute_pgm_rsrc2 | (lds_size << 15)
            return (bytes(sec.content), tuple(lowered.prg.p.global_size), tuple(lowered.prg.p.local_size), buf_sizes, buf_data, rsrc2)
    return None
  except Exception as e:
    print(f"  Error getting kernel: {e}")
    return None

TINYGRAD_TESTS = ["add", "mul", "reduce_sum", "softmax", "exp", "sin", "gelu", "matmul_small"]

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Benchmark RDNA3 emulators")
  parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per benchmark")
  parser.add_argument("--profile", type=str, default=None, help="Profile instructions for a specific kernel (e.g. 'sin')")
  parser.add_argument("--top", type=int, default=20, help="Number of top instructions to show in profile")
  parser.add_argument("--sort-build", action="store_true", help="Sort profile by build time instead of render time")
  args = parser.parse_args()

  # Profile mode: show individual instruction timing
  if args.profile:
    kernel_info = get_tinygrad_kernel(args.profile)
    if kernel_info is None:
      print(f"Failed to get kernel for '{args.profile}'")
      return
    kernel = kernel_info[0]
    print(f"Profiling instructions for '{args.profile}' kernel...")
    print("=" * 140)
    results = profile_instructions(kernel)
    if args.sort_build:
      results = sorted(results, key=lambda x: x['build_ms'], reverse=True)
    print(f"{'Instruction':<90} {'UOps':>6}  {'Build(ms)':>10}  {'Render(ms)':>10}")
    print("-" * 140)
    for r in results[:args.top]:
      inst = r['inst_str'][:87] + "..." if len(r['inst_str']) > 90 else r['inst_str']
      print(f"{inst:<90} {r['uop_count']:>6}  {r['build_ms']:>10.3f}  {r['render_ms']:>10.3f}")
    print("-" * 140)
    total_build = sum(r['build_ms'] for r in results)
    total_render = sum(r['render_ms'] for r in results)
    print(f"{'TOTAL':<90} {'':>6}  {total_build:>10.3f}  {total_render:>10.3f}")
    return

  rust_remu = get_rust_remu()
  if rust_remu is None:
    print("Rust libremu not found. Build with: cargo build --release --manifest-path extra/remu/Cargo.toml")
    print("Running Python-only benchmarks...\n")

  print("=" * 90)
  print("RDNA3 Emulator Benchmark: Python vs Rust")
  print("=" * 90)

  results = []

  print("\n[TINYGRAD KERNELS]")
  print("-" * 90)

  for op_name in TINYGRAD_TESTS:
    print(f"\n{op_name}:", end=" ", flush=True)
    kernel_info = get_tinygrad_kernel(op_name)
    if kernel_info is None:
      print("failed to compile")
      continue

    kernel, global_size, local_size, buf_sizes, buf_data, rsrc2 = kernel_info
    buffers, args_arr, args_ptr, ranges = setup_buffers(buf_sizes, buf_data)

    # Benchmark Python emulator (must be first to measure compile time before cache is populated)
    py_build, py_render, render_count, py_compile, py_exec = benchmark_python_split(kernel, global_size, local_size, args_ptr, rsrc2, args.iterations)

    n_insts = count_instructions(kernel)  # uses cached decode_program
    n_workgroups = global_size[0] * global_size[1] * global_size[2]
    n_threads = local_size[0] * local_size[1] * local_size[2]
    total_work = n_insts * n_workgroups * n_threads

    print(f"{n_insts} insts × {n_workgroups} WGs × {n_threads} threads = {total_work:,} ops")
    rust_time = benchmark_emulator("Rust", rust_remu.run_asm, kernel, global_size, local_size, args_ptr, rsrc2, args.iterations) if rust_remu else None

    if py_build is not None:
      py_exec_rate = total_work / py_exec / 1e6
      print(f"  Build:          {py_build*1000:8.3f} ms")
      print(f"  Render:         {py_render*1000:8.3f} ms  ({render_count} unique)")
      print(f"  Compile:        {py_compile*1000:8.3f} ms")
      print(f"  Exec:           {py_exec*1000:8.3f} ms  ({py_exec_rate:7.2f} M ops/s)")
    if rust_time:
      rust_rate = total_work / rust_time / 1e6
      speedup = py_exec / rust_time if py_exec else 0
      print(f"  Rust:           {rust_time*1000:8.3f} ms  ({rust_rate:7.2f} M ops/s)  [{speedup:.1f}x faster]")

    results.append((op_name, n_insts, n_workgroups, py_build, py_render, render_count, py_compile, py_exec, rust_time))

  # Summary table
  print("\n" + "=" * 140)
  print("SUMMARY")
  print("=" * 140)
  print(f"{'Name':<16} {'Insts':<6} {'WGs':<5} {'Build (ms)':<12} {'Render (ms)':<16} {'Compile (ms)':<14} {'Exec (ms)':<12} {'Rust (ms)':<12} {'Speedup':<10}")
  print("-" * 140)

  for name, n_insts, n_wgs, py_build, py_render, render_count, py_compile, py_exec, rust_time in results:
    build_ms = f"{py_build*1000:.3f}" if py_build else "error"
    render_ms = f"{py_render*1000:.3f} ({render_count})" if py_render else "error"
    compile_ms = f"{py_compile*1000:.3f}" if py_compile else "error"
    exec_ms = f"{py_exec*1000:.3f}" if py_exec else "error"
    if rust_time:
      rust_ms = f"{rust_time*1000:.3f}"
      speedup = f"{py_exec/rust_time:.1f}x" if py_exec else "N/A"
    else:
      rust_ms, speedup = "N/A", "N/A"
    print(f"{name:<16} {n_insts:<6} {n_wgs:<5} {build_ms:<12} {render_ms:<16} {compile_ms:<14} {exec_ms:<12} {rust_ms:<12} {speedup:<10}")

if __name__ == "__main__":
  main()
