import argparse, time
from tinygrad.llm.model import Transformer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="path to gguf model")
  parser.add_argument("--max-context", type=int, default=8192, help="max context length (default: %(default)s)")
  parser.add_argument("--prompt-tokens", type=int, default=1024, help="number of prompt tokens (default: %(default)s)")
  parser.add_argument("--decode-tokens", type=int, default=16, help="number of tokens to decode (default: %(default)s)")
  parser.add_argument("--chunk-size", type=int, default=32, help="chunk size for prefill (default: %(default)s)")
  args = parser.parse_args()

  st = time.perf_counter()
  model, _ = Transformer.from_gguf(args.model, args.max_context)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)

  st = time.perf_counter()
  model.warmup()
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)

  prompt = [257] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  gen = model.generate(prompt, chunk_size=args.chunk_size)
  st = time.perf_counter()
  # first token is time-to-first-token; counted as part of prefill
  output = [next(gen)]
  pt = time.perf_counter()
  print(f"prefill {args.prompt_tokens/(pt-st):.3f} tok/s", flush=True)

  for _ in range(args.decode_tokens): output.append(next(gen))
  et = time.perf_counter()
  print(f"decode {args.decode_tokens/(et-pt):.3f} tok/s output {output}", flush=True)
