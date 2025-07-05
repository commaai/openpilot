#!/usr/bin/env python3
import os, sys, traceback
sys.path.append(os.getcwd())

from io import StringIO
from contextlib import redirect_stdout
from tinygrad import Tensor, nn, Device, dtypes
from tinygrad.helpers import Timing, colored, getenv, fetch
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from sentencepiece import SentencePieceProcessor

def create_fixed_tokenizer(output_file):
  print("creating fixed tokenizer")
  import extra.junk.sentencepiece_model_pb2 as spb2
  mp = spb2.ModelProto()
  mp.ParseFromString(fetch("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/tokenizer.model?download=true").read_bytes())
  mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_end|>", score=0))
  mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_start|>", score=0))
  with open(output_file, "wb") as f:
    f.write(mp.SerializeToString())

# example:
# echo -en "write 2+2\nwrite hello world\ny\n" | TEMP=0 python3 examples/coder.py

if __name__ == "__main__":
  # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/config.json
  with Timing("create model: "):
    model = Transformer(4096, 14336, n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=32002, n_kv_heads=8, max_context=4096, jit=getenv("JIT", 1))

  with Timing("download weights: "):
    part1 = nn.state.torch_load(fetch("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/pytorch_model-00001-of-00002.bin?download=true"))
    part2 = nn.state.torch_load(fetch("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/pytorch_model-00002-of-00002.bin?download=true"))

  with Timing("weights -> model: "):
    nn.state.load_state_dict(model, fix_bf16(convert_from_huggingface(part1, 32, 32, 8)), strict=False)
    nn.state.load_state_dict(model, fix_bf16(convert_from_huggingface(part2, 32, 32, 8)), strict=False)

  if not os.path.isfile("/tmp/tokenizer.model"): create_fixed_tokenizer("/tmp/tokenizer.model")
  spp = SentencePieceProcessor(model_file="/tmp/tokenizer.model")

  # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/tokenizer_config.json
  #   "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
  IM_END = 32000
  IM_START = 32001
  def encode_prompt(k, v): return [IM_START]+spp.encode(f"{k}\n{v}")+[IM_END]+spp.encode("\n")
  def start_prompt(k): return [IM_START]+spp.encode(f"{k}\n")
  def output(outputted, toks, color):
    cur = spp.decode(toks)[len(outputted):]
    sys.stdout.write(colored(cur, color))
    sys.stdout.flush()
    outputted += cur
    return outputted

  # *** app below this line ***

  toks = [spp.bos_id()] + encode_prompt("system", "You are Quentin. Quentin is a useful assistant who writes Python code to answer questions. He keeps the code as short as possible and doesn't read from user input")

  PROMPT = getenv("PROMPT", 1)
  temperature = getenv("TEMP", 0.7)

  start_pos = 0
  outputted = output("", toks, "green")
  turn = True
  while 1:
    if PROMPT:
      toks += encode_prompt("user", input("Q: ")) + start_prompt("assistant")
    else:
      toks += start_prompt("user" if turn else "assistant")
      turn = not turn
    old_output_len = len(outputted)
    while 1:
      tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).item()
      start_pos = len(toks)
      toks.append(tok)
      outputted = output(outputted, toks, "blue" if not turn else "cyan")
      if tok == IM_END: break
      if tok == spp.eos_id(): break
      new_output = outputted[old_output_len:]

      if new_output.endswith("```") and '```python\n' in new_output:
        python_code = new_output.split('```python\n')[1].split("```")[0]
        # AI safety. Warning to user. Do not press y if the AI is trying to do unsafe things.
        if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
          my_stdout = StringIO()
          try:
            with redirect_stdout(my_stdout): exec(python_code)
            result = my_stdout.getvalue()
          except Exception as e:
            result = ''.join(traceback.format_exception_only(e))
          toks += spp.encode(f"\nOutput:\n```\n{result}```")
          outputted = output(outputted, toks, "yellow")
          old_output_len = len(outputted)
    print("")