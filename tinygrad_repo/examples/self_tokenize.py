import os, pathlib
from examples.llama3 import Tokenizer
from tabulate import tabulate
from tinygrad import fetch
from tinygrad.helpers import flatten

# llama 3 tokenizer
tokenizer = Tokenizer(fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model").as_posix())

def read_code(base_path):
  ret = []
  for path, _, files in os.walk(os.path.join(base_path, "tinygrad")):
    for name in files:
      if not name.endswith(".py"): continue
      if 'tinygrad/runtime/autogen' in path.replace('\\', '/'): continue
      fullpath = os.path.join(path, name)
      code = pathlib.Path(fullpath).read_text()
      ret += [(fullpath.split("tinygrad/", 1)[1], code)]
  return ret

if __name__ == "__main__":
  ret = read_code(".")

  table = []
  for name,code in ret:
    table.append([name, len(tokenizer.encode(name+"\x00"+code))])
  print(tabulate([["name", "llm tokens"]]+sorted(table, key=lambda x: -x[1]), headers="firstrow"))

  code_str = '\x00'.join(flatten(ret))
  print(f"code has {len(code_str)} chars")
  newline_count = code_str.count('\n')
  print(f"code has {newline_count} newlines")

  encoded = tokenizer.encode(code_str)
  print(f"code has {len(encoded)} tokens")
