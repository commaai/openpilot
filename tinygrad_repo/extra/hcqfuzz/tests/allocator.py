from extra.hcqfuzz.spec import TestSpec
import random

class TLSFAllocator(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
        "SEED": seed,
        "ITERS": random.randint(10000, 1000000),
    }

    self.cmd = "python3 test/external/external_fuzz_tlsf.py"
    self.timeout = 60 * 60 # 60 minutes

  def get_exec_state(self): return self.env, self.cmd, self.timeout
