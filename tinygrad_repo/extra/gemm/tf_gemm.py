import time
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

for dtype in [tf.float16, tf.float32]:
  for N in [256, 512, 1024, 2048, 4096, 8192]:
    FLOPS = N*N*N*2

    b = tf.random.uniform((N, N), dtype=dtype)
    c = tf.random.uniform((N, N), dtype=dtype)

    b = tf.Variable(b)
    c = tf.Variable(c)

    def tf_prog(b, c):
      st = time.perf_counter()
      a = tf.matmul(b, c)
      tf.debugging.check_numerics(a, "Nan or Inf in result") # Ensures that the calculation is done.
      return time.perf_counter() - st

    tm = min([tf_prog(b, c) for _ in range(20)])
    print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS {N:4d}x{N:4d}x{N:4d} matmul in {dtype}")