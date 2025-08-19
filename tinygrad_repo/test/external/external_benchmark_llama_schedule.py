from tinygrad import nn, Tensor, Device, dtypes
from tinygrad.helpers import Timing

from extra.models.llama import Transformer
from examples.llama3 import MODEL_PARAMS

if __name__ == "__main__":
  Device.DEFAULT = "NULL"
  Tensor.training = True
  #model_size = "8B"
  model_size = "405B"

  with Timing("total "):
    with Timing("***** create model in    "):
      model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=nn.Linear, embedding=nn.Embedding,
                          max_context=1024, jit=True, disable_kv_cache=True)

    with Timing("***** fake state in      "):
      Tensor.realize(*[p.assign(Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)) for p in nn.state.get_parameters(model)])

    with Timing("***** create optim in    "):
      opt = nn.optim.AdamW(nn.state.get_parameters(model))

    with Timing("***** run model in       "):
      toks = Tensor.empty(1, 1024, dtype=dtypes.int)
      out = model(toks, 0, temperature=float('nan'))

    with Timing("***** backward in        "):
      out.mean().backward()

    with Timing("***** realize in        "):
      out.realize()

    with Timing("***** step in           "):
      opt.step()
