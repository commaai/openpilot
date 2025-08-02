from pathlib import Path
from typing import List
import json, argparse, random, time, os
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from extra.models.llama import Transformer, convert_from_huggingface, convert_from_gguf, fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm
from extra.bench_log import BenchEvent, WallTimeEvent

class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])
  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight")) else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".gguf"):
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    return gguf_load(gguf_tensor)[1]
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)

# **** quantized linears ****
class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(self.scale.dtype).T*self.scale)

  @staticmethod
  def quantize(tensors, device, scale_dtype=dtypes.float16, quantize_embeds=False):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name or (quantize_embeds and "tok_embeddings.weight" in name):
        assert "weight" in name, name
        v = v.cast(scale_dtype)
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.round().cast(dtype=dtypes.int8) # without round(), cast truncates -34.9 to -34
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    if quantize_embeds: new_tensors.update({"output.weight": new_tensors["tok_embeddings.weight"], "output.scale": new_tensors["tok_embeddings.scale"]})
    return new_tensors

class Int8Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight, self.scale = Tensor.ones(vocab_size, embed_size, dtype=dtypes.int8), Tensor.ones(vocab_size, dtype=dtypes.half)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), (self.weight.cast(self.scale.dtype).T*self.scale).T
    return (arange == idx).mul(vals).sum(-2, dtype=vals.dtype)

def NF4Linear(block_size):
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor.stack(*[Tensor(c, dtype=dtypes.float16) for c in _CODE])
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits = self.weight
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack(high_bits, low_bits, dim=-1).idiv(2 ** 4)
      unscaled = CODE[unpacked].to(x.device).reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16, quantize_embeds=False) -> dict[str, Tensor]:
      assert not quantize_embeds  # TODO: support this?
      new_state_dict = {}
      for k, v in state_dict.items():
        if "feed_forward" in k or "attention.w" in k:
          grouped = v.reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(scale_dtype)
          if isinstance(device, tuple):
            new_state_dict[k].shard_(device, axis=-1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device, axis=None)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear

MODEL_PARAMS = {
  "1B": {
    "args": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
    "files": 1
  },
  "8B": {
    "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
    "files": 1
  },
  "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
    "files": 8
  },
  "405B": {
    "args": {"dim": 16384, "n_heads": 128, "n_kv_heads": 8, "n_layers": 126, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 53248},
    "files": 191
  },
}
def build_transformer(model_path: Path, model_size="8B", quantize=None, scale_dtype=dtypes.float16, device=None, max_context=8192, load_weights=True):
  # build model
  if quantize == "int8": linear, embedding, quantize_embeds = Int8Linear, Int8Embedding, True
  elif quantize == "nf4": linear, embedding, quantize_embeds = NF4Linear(64), nn.Embedding, False
  else: linear, embedding, quantize_embeds = nn.Linear, nn.Embedding, False
  model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, embedding=embedding, max_context=max_context, jit=True)

  if not load_weights: return model

  # load weights
  with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
    if model_path.is_dir():
      if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
      elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
      else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
    else:
      weights = load(str(model_path))
    if "model.embed_tokens.weight" in weights:
      weights = convert_from_huggingface(weights, MODEL_PARAMS[model_size]["args"]["n_layers"], MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
    elif "token_embd.weight" in weights:
      weights = convert_from_gguf(weights, MODEL_PARAMS[model_size]["args"]["n_layers"])
    weights = fix_bf16(weights)

    with Context(BEAM=0):
      # quantize
      if quantize == "float16": weights = {k:v.cast(quantize).contiguous() for k,v in weights.items()}
      elif quantize is not None:
        weights = linear.quantize(weights, device, scale_dtype, quantize_embeds)
        for _,v in weights.items(): v.realize()

      # shard
      if isinstance(device, tuple):
        for k,v in nn.state.get_state_dict(model).items():
          if 'scale' in k: v.shard_(device, axis=None)  # from quantized
          elif '.attention.' in k: v.shard_(device, axis=-1)
          elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
          elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
          elif '.feed_forward.' in k: v.shard_(device, axis=-1)
          elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
          elif 'output.weight' in k: v.shard_(device, axis=0)
          else: v.shard_(device, axis=None)

      # replace weights in model
      load_state_dict(model, weights, strict=False, consume=True)
  return model

# default settings
TEMPERATURE = 0.95
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

last_seen_toks = []
def prefill(model, toks, start_pos=0):
  global last_seen_toks

  # we can skip part of the prompt if it is the same as last and start_pos=0
  if start_pos == 0:
    for i, (a, b) in enumerate(zip(toks, last_seen_toks)):
      if a != b: break
    else: i = min(len(toks), len(last_seen_toks))
    start_pos += i
    last_seen_toks = toks
    toks = toks[i:]

  # prefill the model
  for tok in tqdm(toks):
    GlobalCounters.reset()
    model(Tensor([[tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
    start_pos += 1
  return start_pos

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--download_model", action="store_true", help="Download a model")
  parser.add_argument("--model", type=Path, help="Model path")
  parser.add_argument("--size", choices=["1B", "8B", "70B", "405B"], default="1B", help="Model size")
  parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
  parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
  parser.add_argument("--no_api", action="store_true", help="Disable the api and run a cli test interface")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
  parser.add_argument("--port", type=int, default=7776, help="Web server port")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
  parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data")
  args = parser.parse_args()

  # download_model is the default without a model passed in
  if args.download_model or not args.model:
    if args.size == "1B":
      fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
      args.model = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
    elif args.size == "8B":
      fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-8b-sfr")
      fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors", "model-00001-of-00004.safetensors", subdir="llama3-8b-sfr")
      fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors", "model-00002-of-00004.safetensors", subdir="llama3-8b-sfr")
      fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors", "model-00003-of-00004.safetensors", subdir="llama3-8b-sfr")
      fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors", "model-00004-of-00004.safetensors", subdir="llama3-8b-sfr")
      args.model = fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json", "model.safetensors.index.json", subdir="llama3-8b-sfr")
    elif args.size == "70B":
      subdir = "DeepSeek-R1-Distill-Llama-70B"
      args.model = fetch("https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json?download=true", "model.safetensors.index.json", subdir=subdir)
      fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir=subdir)
      for i in range(17):
        fetch(f"https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-{i+1:05d}-of-000017.safetensors?download=true", f"model-{i+1:05d}-of-000017.safetensors", subdir=subdir)

  assert args.model is not None, "please provide --model option"

  if args.seed is not None: Tensor.manual_seed(args.seed)
  if args.benchmark: Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  TEMPERATURE = args.temperature

  tokenizer = Tokenizer(str((args.model if args.model.is_dir() else args.model.parent) / "tokenizer.model"))
  def encode_role(role: str):
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
  def encode_message(role: str, content: str):
    return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT
  model = build_transformer(args.model, model_size=args.size, quantize=args.quantize, device=device)
  param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(model))

  if not args.no_api and not args.benchmark:
    from bottle import Bottle, request, response, HTTPResponse, abort, static_file
    app = Bottle()

    cors_headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
      "Access-Control-Allow-Credentials": "true",
    }
    @app.hook("before_request")
    def handle_options():
      if request.method == "OPTIONS": raise HTTPResponse(headers=cors_headers)
    @app.hook("after_request")
    def enable_cors():
      for key, value in cors_headers.items(): response.set_header(key, value)

    @app.route("/<filename>")
    def server_static(filename): return static_file(filename, root=(Path(__file__).parent / "tinychat").as_posix())
    @app.route("/assets/<filename:path>")
    def server_assets(filename): return static_file(filename, root=(Path(__file__).parent / "tinychat" / "assets").as_posix())
    @app.route("/")
    def index():
      return static_file("index.html", root=(Path(__file__).parent / "tinychat").as_posix())

    @app.get("/v1/models")
    def models():
      return json.dumps([str(args.model)])

    @app.post("/v1/internal/token-count")
    def token_count():
      rjson = json.loads(request.body.read())
      return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))
    @app.post("/v1/token/encode")
    def token_encode():
      rjson = json.loads(request.body.read())
      return json.dumps(tokenizer.encode(rjson.get("text", "")))

    @app.post("/v1/completions")
    def completions():
      rjson = json.loads(request.body.read())

      # check if we are streaming
      if rjson.get("stream", False):
        response.content_type = "text/event-stream"
        response.set_header("Cache-Control", "no-cache")
      else: abort(400, "streaming required")

      toks = [tokenizer.bos_id] + tokenizer.encode(rjson.get("prompt", ""), allow_special=True)

      start_pos = prefill(model, toks[:-1])
      last_tok = toks[-1]
      while True:
        GlobalCounters.reset()
        tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
        start_pos += 1
        last_tok = tok
        if tok in tokenizer.stop_tokens: break

        res = {
          "choices": [{
            "text": tokenizer.decode([tok]),
          }]
        }
        yield f"data: {json.dumps(res)}\n\n"

    @app.post("/v1/chat/token/encode")
    def chat_token_encode():
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")
      toks = [tokenizer.bos_id]
      for message in rjson["messages"]:
        toks += encode_message(message["role"], message["content"])
      if len(rjson["messages"]) > 0 and message["role"] == "user":
        toks += encode_role("assistant")
      return json.dumps(toks)

    @app.post("/v1/chat/completions")
    def chat_completions():
      global last_seen_toks
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")

      # check if we are streaming
      if rjson.get("stream", False):
        response.content_type = "text/event-stream"
        response.set_header("Cache-Control", "no-cache")
      else: abort(400, "streaming required")

      toks = [tokenizer.bos_id]
      for message in rjson["messages"]:
        toks += encode_message(message["role"], message["content"])
      # ensure that the last message was a user message
      if message["role"] != "user": abort(400, "last message must be a user message")
      toks += encode_role("assistant")

      random_id = random.randbytes(16).hex()

      start_pos = prefill(model, toks[:-1])
      last_tok = toks[-1]
      last_seen_toks.append(last_tok)
      while True:
        GlobalCounters.reset()
        tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
        start_pos += 1
        last_tok = tok
        last_seen_toks.append(tok)
        if tok in tokenizer.stop_tokens: break

        res = {
          "id": random_id,
          "object": "chat.completion.chunk",
          "created": int(time.time()),
          "model": str(args.model),
          "choices": [{
            "index": 0,
            "delta": {
              "role": "assistant",
              "content": tokenizer.decode([tok]),
            },
            "finish_reason": None,
          }]
        }
        yield f"data: {json.dumps(res)}\n\n"

      res = {
        "id": random_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": str(args.model),
        "choices": [{
          "index": 0,
          "delta": {},
          "finish_reason": "stop",
        }]
      }
      yield f"data: {json.dumps(res)}\n\n"

    app.run(host=args.host, port=args.port, debug=args.debug)
  elif args.benchmark:
    toks = [tokenizer.bos_id] + encode_message("user", "Hello.") + encode_role("assistant")

    start_pos = prefill(model, toks[:-1])
    last_tok = toks[-1]
    generated = ""
    for _ in range(20):
      GlobalCounters.reset()
      st = GlobalCounters.time_sum_s
      with Profiling(enabled=args.profile):
        with Timing("total ", on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
          with WallTimeEvent(BenchEvent.STEP):
            with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                        f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                        (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None):
              tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
            tok = tok.item()
      start_pos += 1
      last_tok = tok
      generated += tokenizer.decode([tok])
      print(generated)
    if "LLaMA-3/8B-SF-DPO" in args.model.as_posix() and (TEMPERATURE == 0.85 or TEMPERATURE == 0):
      if TEMPERATURE == 0.85:
        EXPECTED_TEXT = {
          1: "Hello! How can I help you today? If you have any questions or need assistance with anything,",
          2: "Hello! How can I help you today? If you have any questions, need assistance or just want",
          3: "Hello! How can I help you today? If you have any questions or need assistance, feel free",
          4: "Hello! How can I assist you today? If you have any questions, need information, or require",
          5: "Hello! How can I assist you today? If you have any questions or need help with something",
          6: "Hello! How can I assist you today? If you have any questions, need information, or require",
        }
      else:
        EXPECTED_TEXT = {k: "Hello! How can I assist you today? If you have any questions or need help with something," for k in range(1, 7)}
      assert generated == EXPECTED_TEXT[args.shard], f"{generated=} {EXPECTED_TEXT[args.shard]}"
      print("\n" + colored("output validated", "green"))  # NOTE: "\n" inside colored does not render the color in github action
  else:
    prompt = [tokenizer.bos_id] + encode_message("system", "You are an helpful assistant.")

    start_pos = prefill(model, prompt)
    while True:
      toks = encode_message("user", input("Q: ")) + encode_role("assistant")

      start_pos = prefill(model, toks[:-1], start_pos=start_pos)
      last_tok = toks[-1]
      while True:
        GlobalCounters.reset()
        if args.timing or args.profile: print("")
        st = GlobalCounters.time_sum_s
        with Profiling(enabled=args.profile):
          with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
            with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                        f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                        (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):

              tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
            tok = tok.item()
        start_pos += 1
        last_tok = tok
        if tok in tokenizer.stop_tokens: break
        print(tokenizer.decode([tok]), end="", flush=True)
      print(flush=True)
