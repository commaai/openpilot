import unittest, time

from tinygrad import Tensor, TinyJit, GlobalCounters, Device
from tinygrad.helpers import getenv, Context
from tinygrad.nn.optim import LAMB
from tinygrad.nn.state import get_parameters
from tinygrad.engine.realize import run_schedule

from extra.models import bert

bs = getenv("BS", 16)
seq_len = getenv("SEQ_LEN", 512)

class BenchmarkBertTrain(unittest.TestCase):
  def _get_layer(self, layer_id):
    if not hasattr(self, "model"):
      dropout_prob = 0.0 if getenv("DISABLE_DROPOUT") else 0.1
      self.model = bert.BertForPretraining(attention_probs_dropout_prob=dropout_prob, hidden_dropout_prob=dropout_prob)
    hidden_size = self.model.bert.embeddings.word_embeddings.embed_sz
    intermediate_size = self.model.bert.encoder.layer[0].intermediate.dense.weight.shape[0]

    layer_map = {
      "embedding": self.model.bert.embeddings,
      "attention_self": self.model.bert.encoder.layer[0].attention.self,
      "attention_output": self.model.bert.encoder.layer[0].attention.output,
      "intermediate": self.model.bert.encoder.layer[0].intermediate,
      "output": self.model.bert.encoder.layer[0].output
    }
    input_shapes = {
      "embedding": [(bs, seq_len), (bs, seq_len)],
      "attention_self": [(bs, seq_len, hidden_size), (bs, 1, 1, seq_len)],
      "attention_output": [(bs, seq_len, hidden_size), (bs, seq_len, 1)],
      "intermediate": [(bs, seq_len, hidden_size)],
      "output": [(bs, seq_len, intermediate_size), (bs, seq_len, 1)]
    }.get(layer_id)

    return f"{layer_id}-layer, Input: {input_shapes}", layer_map.get(layer_id), input_shapes

  def _test_layer(self, name, layer, input_shapes):
    optim = LAMB(get_parameters(layer))
    with Context(TRACK_MATCH_STATS=0): Tensor.realize(*[t.assign(t.detach().contiguous()) for t in get_parameters(optim)])

    JITCNT = getenv("JITCNT", 1)
    Tensor.training = True
    @TinyJit
    def step(inputs):
      optim.zero_grad()
      for i in inputs: i.grad = None

      y = layer(*inputs).contiguous().contiguous_backward()
      y.sum().backward()
      if getenv("ASSIGN", 1): sched, _ = Tensor.schedule_with_vars(y, *list(inputs), *optim.schedule_step())
      else: sched, _ = Tensor.schedule_with_vars(y, *list(inputs), *[t.grad for t in optim.params])

      for _ in range(JITCNT):
        run_schedule(sched)

    CNT = getenv("CNT", 5)
    best_tm = None
    flops, mem_used, mem, kernels = None, None, None, None
    for _ in range(CNT):
      with Context(TRACK_MATCH_STATS=0): inputs = [Tensor.randn(*shape, requires_grad=False).realize() for shape in input_shapes]
      GlobalCounters.reset()

      st = time.perf_counter()
      step(inputs)
      Device[Device.DEFAULT].synchronize()
      et = time.perf_counter()

      flops = GlobalCounters.global_ops / JITCNT
      mem_used = GlobalCounters.mem_used
      mem = GlobalCounters.global_mem / JITCNT
      if kernels is None: kernels = GlobalCounters.kernel_count // JITCNT
      tm = (et-st) / JITCNT
      if best_tm is None or tm < best_tm: best_tm = tm
    print(f"\r{name:70s}: {best_tm * 1000:>9.2f} ms, {flops / 10**12 / best_tm:>6.2f} TFLOPS, {mem / 10**9 / best_tm:>5.0f} GB/s, "
          f"{mem_used / 10**9: 6.2f} GB used, {kernels:>5d} kernels")
    return best_tm, flops, mem, kernels

  def test_embedding_layer(self): self._est(*self._test_layer(*self._get_layer("embedding")), 1)
  def test_attention_self_layer(self): self._est(*self._test_layer(*self._get_layer("attention_self")), 24) # Assumes BERT-large
  def test_attention_output_layer(self): self._est(*self._test_layer(*self._get_layer("attention_output")), 24)
  def test_intermediate_layer(self): self._est(*self._test_layer(*self._get_layer("intermediate")), 24)
  def test_output_layer(self): self._est(*self._test_layer(*self._get_layer("output")), 24)

  est_tm, est_flops, est_mem, est_kernels = 0, 0, 0, 0

  @classmethod
  def _est(cls, tm, flops, mem, kernels, mult):
    cls.est_tm += tm * mult
    cls.est_flops += flops * mult
    cls.est_mem += mem * mult
    cls.est_kernels += kernels * mult

  @classmethod
  def tearDownClass(cls):
    print(f"\restimated step tm: {cls.est_tm * 1000.0:.2f} ms, {cls.est_flops / 10 ** 12 / cls.est_tm:.3f} tflops, "
          f"{cls.est_mem / 10 ** 9 / cls.est_tm:.2f} GB/s, {cls.est_kernels} kernels")


if __name__ == "__main__":
  unittest.main()
