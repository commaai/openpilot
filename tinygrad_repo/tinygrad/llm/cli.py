from __future__ import annotations
import sys, argparse, json, time, re
from typing import TYPE_CHECKING
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import DEBUG, Timing, GlobalCounters, Context, fetch, profile_marker, getenv, round_up
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
if TYPE_CHECKING:
  import jinja2

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.6:27b": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf",
  "qwen3.6:35b-a3b": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  # pinned to the last revision with the plain IQ4_XS quant (the replacement uses mixed UD quantization)
  "qwen3.8:27b": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/b62a80264f8b0c1bb849ee1c9c487415ebeca194/Qwen3.8-27B-IQ4_XS.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

class FallbackTemplate:
  # minimal jinja2.Template-compatible chat template without jinja2, no tool calling support
  def __init__(self, tok:SimpleTokenizer): self.tok = tok
  def role(self, role:str) -> str:
    if self.tok.preset == 'olmo': return "<|" + role + "|>\n"  # OLMoE Instruct format
    if self.tok.preset == 'kimi-k2': return "<|im_" + role + "|>" + role + "<|im_middle|>"
    if self.tok.preset == 'qwen2': return "<|im_start|>" + role + "\n"
    if self.tok.preset == 'glm4': return "<|" + role + "|>"
    if self.tok.preset == 'tekken':
      if role == 'user': return "[INST]"
      if role == 'assistant': return ""
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.tok.preset}'")
    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
  def end_turn(self) -> str:
    if self.tok.preset == 'olmo': return "\n"
    if self.tok.preset == 'kimi-k2': return self.tok.decode([self.tok.eos_id])
    if self.tok.preset == 'qwen2': return self.tok.decode([self.tok.eos_id]) + "\n"
    if self.tok.preset == 'glm4': return ""
    if self.tok.preset == 'tekken': return "[/INST]"
    return self.tok.decode([self.tok.eos_id])
  def render(self, messages:list[dict], tools=None, add_generation_prompt:bool=True, preserve_thinking:bool=False) -> str:
    out = self.tok.decode([self.tok.bos_id] if self.tok.add_bos and self.tok.bos_id is not None else [])
    out += "<sop>" if self.tok.preset == 'glm4' else ""
    for msg in messages:
      out += self.role(msg["role"])
      content = msg.get("content")
      if isinstance(content, str): out += content
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": out += c["text"]
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      elif content is not None: raise RuntimeError(f"unknown content type: {type(content)}")
      out += self.end_turn()
    return out + self.role("assistant") if add_generation_prompt else out

from tinygrad.llm.serve import LLMServer

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0],
                      help=f"Model choice ({', '.join(models.keys())}) or local GGUF file/HF checkpoint directory")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=lambda s: tuple(int(x) for x in (s if "," in s else f"0,{s}").split(",", 1)), const=(0, 20),
                      metavar="[PREFILL,]DECODE", help="Benchmark tok/s (PREFILL prompt tokens then DECODE new tokens, default 0,20)")
  parser.add_argument("--no_chat_template", action="store_true", help="Don't use the model's chat template, always use the fallback template")
  parser.add_argument("--shard", type=int, default=1, help="Tensor parallel device count")
  args = parser.parse_args()
  if args.benchmark: args.max_context = max(args.max_context, round_up(sum(args.benchmark)+1, 1024))

  # load the model
  st = time.perf_counter()
  with Context(DEBUG=max(DEBUG.value, 1 if args.serve else 0)):
    path = fetch(models.get(args.model, args.model))
    model, kv = (Transformer.from_hf if path.is_dir() else Transformer.from_gguf)(path, args.max_context, shard=args.shard)
  model_name = kv.get('general.name') or kv.get('general.basename') or kv.get('_name_or_path') or args.model
  file_sizes = [y.nbytes()*args.shard for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"loaded model \"{model_name}\" at {sum(file_sizes)*1e-9/(time.perf_counter()-st):.2f} GB/s with {sum(file_sizes):,} bytes "
        f"and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params, "
        f"max context {args.max_context} on {nn.state.get_parameters(model)[0].device}")

  # get tokenizer
  tok = SimpleTokenizer.from_hf(path, preset='glm4') if path.is_dir() else SimpleTokenizer.from_gguf_kv(kv)

  # use the model's chat template if jinja2 is available (enables model-specific formatting)
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if not args.no_chat_template and (ct := tok.chat_template) is not None:
    try:
      import jinja2
      # {% generation %} only marks training spans, keep it as a plain block
      ct = re.sub(r"(\{%-?\s*)(end)?generation\b", lambda m: m[1] + ("endif" if m[2] else "if true"), ct)
      env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])  # (for {% break %})
      env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)  # jinja2's tojson escapes <>& for HTML safety
      env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
      env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
      env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
      env.globals['eos_token'] = tok.decode([tok.eos_id])
      template = env.from_string(ct)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")

  # warmup the JIT
  if args.warmup or args.serve:
    with Context(DEBUG=max(DEBUG.value, 1)): model.warmup()

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok, template).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    keep, (n_pp, n_tg) = 20, args.benchmark
    gen = model.generate(toks:=tok.encode(" ".join(map(str, range(n_pp))))[:n_pp] if n_pp else [tok.bos_id or 0])
    for i in range(n_tg+1):
      profile_marker(f"decode @ {i-1}" if i else "prefill")
      if i == keep+1 and n_tg > 2*keep: print("...")
      GlobalCounters.reset()
      if (log:=getenv("BENCHMARK_LOG", "")): from extra.bench_log import WallTimeEvent, BenchEvent
      with Timing(on_exit=lambda x: f", {(max(n_pp, 1) if i == 0 else 1)*1e9/x:8.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks[n_pp if i <= keep or n_tg <= 2*keep else max(n_pp, 1)+n_tg-keep+1:]).replace("\n", "\\n"),
                  prefix=f"prefill {max(n_pp, 1):5d}: " if i == 0 else f"decode {i-1:6d}: ", enabled=i <= keep or i > n_tg-keep):
        if log:
          with WallTimeEvent(BenchEvent.STEP): next(gen)
        else: next(gen)
    exit(0)

  # interactive chat
  messages: list[dict] = []
  while 1:
    try: messages.append({"role":"user", "content":input('>>> ')})
    except EOFError: break
    ids = tok.encode(template.render(messages=messages, add_generation_prompt=True))
    reply, dec = "", tok.stream_decoder()
    for next_id in model.generate(ids):
      if tok.is_end(next_id):
        sys.stdout.write(dec() + "\n\n")
        break
      reply += (piece := dec(next_id))
      sys.stdout.write(piece)
      sys.stdout.flush()
    messages.append({"role":"assistant", "content":reply})

if __name__ == "__main__": main()
