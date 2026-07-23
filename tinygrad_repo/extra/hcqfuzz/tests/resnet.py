from extra.hcqfuzz.spec import TestSpec
import random

resnet_train_params = {
  "DEFAULT_FLOAT": "HALF",
  "SUM_DTYPE": "HALF",
  "GPUS": 6,
  "BS": 1536,
  "EVAL_BS": 192,
  "TRAIN_BEAM": 4,
  "IGNORE_JIT_FIRST_BEAM": 1,
  "BEAM_UOPS_MAX": 2000,
  "BEAM_UPCAST_MAX": 96,
  "BEAM_LOCAL_MAX": 1024,
  "BEAM_MIN_PROGRESS": 5,
  "BEAM_PADTO": 0,
  "EVAL_START_EPOCH": 3,
  "EVAL_FREQ": 4
}

class TrainResnet(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      **resnet_train_params,
      "IGNORE_BEAM_CACHE": 1,
      "SEED": seed,
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 4 * 60 * 60 # 7 hours

  def get_exec_state(self): return self.env, self.cmd, self.timeout

class TrainResnetShort(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      **resnet_train_params,
      "SEED": seed,
      "BENCHMARK": 4096,
      "JIT": 2,
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 2 * 60 * 60 # 2 hours

  def get_exec_state(self): return self.env, self.cmd, self.timeout

class ResnetBeam(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      **resnet_train_params,
      "IGNORE_BEAM_CACHE": 1,
      "BENCHMARK": 10,
      "SEED": seed,
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 1 * 60 * 60 # 1 hour

  def get_exec_state(self): return self.env, self.cmd, self.timeout
