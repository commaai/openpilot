#!/usr/bin/env python3
import time
import jax
import jax.numpy as jnp

print(jax.devices())
DEVICES = len(jax.devices())
BS = 32
N = 4096
dtype = jnp.float16
A = jnp.zeros((DEVICES, BS, N, N), dtype)
B = jnp.zeros((1, 1, N, N), dtype)
A = jax.device_put_sharded([A[i] for i in range(DEVICES)], jax.devices())
B = jax.device_put_sharded([B for i in range(DEVICES)], jax.devices())

OPS = DEVICES*BS*N*N*N*2
def matmul(A,B): return jnp.matmul(A,B,preferred_element_type=jnp.float32)
pmatmul = jax.pmap(matmul)

MAX_TFLOPS = 123*DEVICES  # Peak FP16 Tensor TFLOPS with FP32 Acc (7900XTX)
for i in range(10):
  st = time.perf_counter()
  C = pmatmul(A,B).block_until_ready()
  et = time.perf_counter()-st
  tflops = (OPS*1e-12)/et
  print(f"time {et*1e3:.2f} ms, TFLOPS {tflops:6.2f}, MFU {(tflops/MAX_TFLOPS)*100:4.2f}% out shape {C.shape} dtype {C.dtype}")

