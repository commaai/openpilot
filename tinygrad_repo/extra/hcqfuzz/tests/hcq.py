from extra.hcqfuzz.spec import TestSpec
import random

class HCQSignalFuzzer(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      "GPUS": random.choice([2, 3, 4, 5, 6]),
      "ITERS": random.randint(1000000, 10000000),
      "SEED": seed,
    }

    self.cmd = "python3 test/external/external_fuzz_hcq_signals.py"
    self.timeout = 30 * 60 # 30 minutes

  def get_exec_state(self): return self.env, self.cmd, self.timeout

class HCQGraphFuzzer(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      "FUZZ_GRAPH_SPLIT_RUNS": random.randint(48, 64),
      "FUZZ_GRAPH_MAX_SPLITS": random.randint(4, 16),
      "FUZZ_GRAPH_SPLIT_RETRY_RUNS": random.randint(4, 8),
      "MAX_KERNELS": random.randint(32, 512),
      "MAX_DEVICES": random.choice([2, 3, 4, 5, 6]),
      "ITERS": random.randint(100, 1000),
    }

    self.cmd = "python3 test/external/fuzz_graph.py"
    self.timeout = 60 * 60 # 60 minutes

  def get_exec_state(self): return self.env, self.cmd, self.timeout
