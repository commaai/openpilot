from extra.hcqfuzz.spec import TestSpec
import random

bert_train_params = {
  "DEFAULT_FLOAT": "HALF",
  "SUM_DTYPE": "HALF",
  "GPUS": 6,
  "BS": 96,
  "EVAL_BS": 96,
  "FUSE_ARANGE": 1,
  "FUSE_ARANGE_UINT": 0,
  "BASEDIR": "/raid/datasets/wiki",
}

class TrainBert(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      **bert_train_params,
      "IGNORE_BEAM_CACHE": 1,
      "BEAM": 5,
      "BEAM_UOPS_MAX": 10000,
      "BEAM_UPCAST_MAX": 256,
      "BEAM_LOCAL_MAX": 1024,
      "BEAM_MIN_PROGRESS": 5,
      "IGNORE_JIT_FIRST_BEAM": 1,
      "LOGMLPERF": 0,
      "SEED": seed,
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 7 * 60 * 60 # 7 hours

  def get_exec_state(self): return self.env, self.cmd, self.timeout

class TrainBertShort(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      **bert_train_params,
      "IGNORE_BEAM_CACHE": 1,
      "BEAM": 5,
      "BEAM_UOPS_MAX": 10000,
      "BEAM_UPCAST_MAX": 256,
      "BEAM_LOCAL_MAX": 1024,
      "BEAM_MIN_PROGRESS": 5,
      "IGNORE_JIT_FIRST_BEAM": 1,
      "SEED": seed,
      "BENCHMARK": 4096,
      "JIT": 2
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 2 * 60 * 60 # 2 hours

  def get_exec_state(self): return self.env, self.cmd, self.timeout

class BertBeam(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      **bert_train_params,
      "IGNORE_BEAM_CACHE": 1,
      "BEAM": random.choice([1, 2, 3, 4, 5]),
      "BEAM_UOPS_MAX": 10000,
      "BEAM_UPCAST_MAX": 256,
      "BEAM_LOCAL_MAX": 1024,
      "BEAM_MIN_PROGRESS": 5,
      "IGNORE_JIT_FIRST_BEAM": 1,
      "SEED": seed,
      "RESET_STEP": 1,
      "BENCHMARK": 10,
      "BERT_LAYERS": 2,
      "SEED": seed,
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 1 * 60 * 60 # 1 hour

  def get_exec_state(self): return self.env, self.cmd, self.timeout
