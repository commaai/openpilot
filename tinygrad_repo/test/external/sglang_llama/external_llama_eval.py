from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks import TaskManager
from pathlib import Path
import json, argparse

from examples.llama3 import build_transformer, Tokenizer, MODEL_PARAMS
from tinygrad import Tensor, Device
from tinygrad.helpers import tqdm

class LLaMaAdaptor(LM):
  def __init__(
    self,
    model_size: str,
    checkpoint_path: Path,
    max_length: int,
    quantize: str | None,
  ):
    super().__init__()
    self.max_length = max_length
    self.tokenizer = Tokenizer(str((checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent) / "tokenizer.model"))
    self.model = build_transformer(checkpoint_path, model_size=model_size, quantize=quantize, max_context=self.max_length)
    self.last_seen_toks = []
  def _prefill(self, toks, temperature) -> int:
    start_pos = 0
    # we can skip part of the prompt if it is the same as last
    for i, (a, b) in enumerate(zip(toks, self.last_seen_toks)):
      if a != b: break
    else: i = min(len(toks), len(self.last_seen_toks))
    start_pos += i
    self.last_seen_toks = toks
    toks = toks[i:]

    # prefill the model
    for tok in toks:
      self.model(Tensor([[tok]]), start_pos, temperature).realize()
      start_pos += 1
    return start_pos

  @property
  def tokenizer_name(self) -> str: pass
  def chat_template(self, chat_template: bool | str = False) -> str: pass
  def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    ret = ""
    for message in chat_history:
      ret += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"
    if add_generation_prompt: ret += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return ret

  def generate_until(self, requests: list[Instance]) -> list[str]:
    continuations = []
    for request in tqdm(requests):
      prompt, args = request.args
      until = [self.tokenizer.encode(tok) for tok in args.get("until", [])]
      toks = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt,allow_special=True)
      prompt_len = len(toks)
      max_gen_toks = args.get("max_gen_toks") or args.get("max_length") or self.max_length-prompt_len
      assert self.max_length >= max_gen_toks, "This eval needs a longer context length"
      temperature = args.get("temperature", 0.0)
      start_pos = self._prefill(toks[:-1], temperature)

      for _ in range(max_gen_toks):
        next_tok = self.model(Tensor([toks[start_pos:]]), start_pos, temperature).item()
        if next_tok in self.tokenizer.stop_tokens or next_tok in until: break
        toks.append(next_tok)
        start_pos += 1

      continuations.append(self.tokenizer.decode(toks[prompt_len:]))
    return continuations

  def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]: raise NotImplementedError() # needs changes to extra/models/llama.py
  def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]: raise NotImplementedError()

if __name__ == '__main__':
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run LLaMA evals in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--size', type=str, default="8B", help=f"Size of model to use [{', '.join(list(MODEL_PARAMS.keys()))}]")
  parser.add_argument('--chat', action='store_true', help="Use chat model")
  parser.add_argument('--ctx', type=int, default=8192, help="Max context length")
  parser.add_argument('--quantize', type=str, default=None, help="Quantize the weights to int8 or int4 in memory")
  parser.add_argument('--eval', type=str, default="mgsm_en_cot_sglang", help="Run in evaluation mode")
  parser.add_argument('--limit', type=int, default=None, help="Limit tests in eval")
  parser.add_argument('--num_fewshot', type=int, default=None, help="Number of examples to add to context")
  parser.add_argument('--model', type=Path, default="./weights/LLaMa/", help="Location of the weights")
  parser.add_argument('--output_path', type=Path, default=None, help="Location of the log file")
  args = parser.parse_args()

  # run eval and exit
  adaptor = LLaMaAdaptor(model_size=args.size, quantize=args.quantize,
                         checkpoint_path=args.model, max_length=args.ctx)
  task_manager = TaskManager(include_path="./")
  results = simple_evaluate(model=adaptor, tasks=args.eval.split(","), task_manager=task_manager, apply_chat_template=args.chat,
                            num_fewshot=args.num_fewshot, limit=args.limit)

  if args.output_path: args.output_path.write_text(json.dumps(results, indent=2))
  for task_name, val in results["results"].items():
    print(f"{task_name}:")
    print("\n".join(f"\t{k}: {v}" for k, v in val.items() if k != "alias"))
