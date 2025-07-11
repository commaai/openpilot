from extra.hcqfuzz.spec import TestSpec
import random

class RingAllreduce(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      "GPUS": random.choice([2, 3, 4, 5, 6]),
      "ITERS": random.randint(10, 1000),
      "DEBUG": 2,
    }

    self.cmd = "python3 test/external/external_benchmark_multitensor_allreduce.py"
    self.timeout = 10 * 60 # 10 minutes

  def get_exec_state(self): return self.env, self.cmd, self.timeout
