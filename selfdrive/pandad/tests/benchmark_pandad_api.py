#!/usr/bin/env python3
"""Benchmark for pandad API implementations."""

import random
import statistics
import time

# Number of iterations and message counts
ITERATIONS = 100
MSG_COUNTS = [10, 100, 500, 1000]


def generate_can_messages(count):
  """Generate random CAN messages for benchmarking."""
  return [
    (random.randint(1, 0x1FFFFFFF),  # address (29-bit)
     bytes(random.getrandbits(8) for _ in range(random.randint(1, 8))),  # data
     random.randint(0, 2))  # src bus
    for _ in range(count)
  ]


def benchmark_serialization(impl_func, msgs, iterations):
  """Benchmark serialization function."""
  times = []
  for _ in range(iterations):
    start = time.perf_counter()
    result = impl_func(msgs)
    end = time.perf_counter()
    times.append(end - start)
  return result, times


def benchmark_deserialization(impl_func, data, iterations):
  """Benchmark deserialization function."""
  times = []
  for _ in range(iterations):
    start = time.perf_counter()
    result = impl_func((data,))
    end = time.perf_counter()
    times.append(end - start)
  return result, times


def print_stats(name, times):
  """Print benchmark statistics."""
  print(f"  {name}:")
  print(f"    Mean:   {statistics.mean(times)*1000:.3f} ms")
  print(f"    Median: {statistics.median(times)*1000:.3f} ms")
  print(f"    Stdev:  {statistics.stdev(times)*1000:.3f} ms")
  print(f"    Min:    {min(times)*1000:.3f} ms")
  print(f"    Max:    {max(times)*1000:.3f} ms")


def verify_roundtrip(serialize, deserialize, msgs):
  """Verify round-trip correctness."""
  data = serialize(msgs)
  result = deserialize((data,))

  assert len(result) == 1, f"Expected 1 result, got {len(result)}"
  _, frames = result[0]

  assert len(frames) == len(msgs), f"Expected {len(msgs)} frames, got {len(frames)}"

  for i, (orig, decoded) in enumerate(zip(msgs, frames)):
    orig_addr, orig_dat, orig_src = orig
    dec_addr, dec_dat, dec_src = decoded

    assert orig_addr == dec_addr, f"Frame {i}: address mismatch {orig_addr} != {dec_addr}"
    assert orig_dat == bytes(dec_dat), f"Frame {i}: data mismatch"
    assert orig_src == dec_src, f"Frame {i}: src mismatch {orig_src} != {dec_src}"


def main():
  from openpilot.selfdrive.pandad.pandad_api_impl import (
    can_list_to_can_capnp as python_serialize,
    can_capnp_to_list as python_deserialize
  )

  print(f"Benchmark: {ITERATIONS} iterations per test\n")

  # Verify correctness first
  print("Verifying round-trip correctness...")
  test_msgs = generate_can_messages(100)
  verify_roundtrip(python_serialize, python_deserialize, test_msgs)

  # Also test with sendcan msgtype
  data = python_serialize(test_msgs, msgtype='sendcan')
  result = python_deserialize((data,), msgtype='sendcan')
  assert len(result) == 1
  print("Round-trip verification passed!\n")

  for msg_count in MSG_COUNTS:
    print(f"\n{'='*50}")
    print(f"Message count: {msg_count}")
    print('='*50)

    msgs = generate_can_messages(msg_count)

    # Benchmark serialization
    print("\nSerialization (can_list_to_can_capnp):")
    py_data, py_times = benchmark_serialization(python_serialize, msgs, ITERATIONS)
    print_stats("Pure Python", py_times)
    print(f"    Throughput: {msg_count / statistics.mean(py_times) / 1000:.1f}k msgs/sec")

    # Benchmark deserialization
    print("\nDeserialization (can_capnp_to_list):")
    py_result, py_times = benchmark_deserialization(python_deserialize, py_data, ITERATIONS)
    print_stats("Pure Python", py_times)
    print(f"    Throughput: {msg_count / statistics.mean(py_times) / 1000:.1f}k msgs/sec")


if __name__ == "__main__":
  main()
