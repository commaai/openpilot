import functools, argparse, pathlib
from tinygrad import Tensor, nn, Device, GlobalCounters, Variable
from tinygrad.helpers import Timing, Profiling, CI, tqdm
from tinygrad.nn.state import torch_load, get_state_dict
from extra.models.llama import FeedForward, Transformer

class MixtureFeedForward:
  def __init__(self, num_experts:int, dim:int, hidden_dim:int, linear=nn.Linear):
    self.gate = nn.Linear(dim, num_experts, bias=False)
    self.experts = [FeedForward(dim, hidden_dim, linear) for _ in range(num_experts)]
  def __call__(self, x:Tensor) -> Tensor:
    assert x.shape[0] == 1, "only BS=1"
    g = self.gate(x).float().exp()
    choice = g.data().tolist()[0][0]
    top = sorted(enumerate(choice), key=lambda x: -x[1])
    norm = top[0][1] + top[1][1]
    e1, e2 = self.experts[top[0][0]], self.experts[top[1][0]]
    scale = Tensor([top[0][1]/norm, top[1][1]/norm])
    ret = e1(x.to(e1.w1.weight.device)).to(x.device) * scale[0] + \
          e2(x.to(e2.w1.weight.device)).to(x.device) * scale[1]
    return ret

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Mixtral in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Profile generation")
  parser.add_argument("--weights", type=str, default=(pathlib.Path(__file__).parent.parent / "weights/mixtral-8x7b-32kseqlen").as_posix(),
                      help="Path to the downloaded weights")
  args = parser.parse_args()

  state = torch_load(args.weights + "/consolidated.00.pth.b")
  model = Transformer(n_layers=32, dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8, norm_eps=1e-5, vocab_size=32000, feed_forward=functools.partial(MixtureFeedForward, 8), jit=False)
  model_state_dict = get_state_dict(model)

  for k in (t := tqdm(state, disable=CI)):
    if 'feed_forward.experts.' in k:
      expert_no = int(k.split('feed_forward.experts.')[1].split('.')[0])
      device = Device.DEFAULT + ":" + str((expert_no//2)+1)
    else:
      device = Device.DEFAULT
    t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, loading {k} to {device}")
    model_state_dict[k].replace(state[k].to(device).half()).realize()
  if CI: print(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")

  from sentencepiece import SentencePieceProcessor
  spp = SentencePieceProcessor(model_file=args.weights + "/tokenizer.model")

  toks = [spp.bos_id()]
  start_pos = 0
  for i in range(args.count):
    GlobalCounters.reset()
    with Profiling(sort="time", frac=0.1, enabled=args.profile):
      with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/sec"):
        tok = model(Tensor([toks[start_pos:]]), 0 if start_pos == 0 else Variable("start_pos", 1, 1024).bind(start_pos), args.temperature).item()
    toks.append(tok)
    start_pos += 1
    print(spp.decode(toks))
