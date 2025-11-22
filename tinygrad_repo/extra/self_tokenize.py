import os, pathlib, argparse
from examples.llama3 import Tokenizer
from tabulate import tabulate
from tinygrad import fetch
from tinygrad.helpers import flatten, getenv
from sz import NONCORE_DIRS

# llama 3 tokenizer
tokenizer = Tokenizer(fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model").as_posix())

def read_code(base_path):
  ret = []
  for path, _, files in os.walk(os.path.join(base_path, "tinygrad")):
    if not getenv("CORE") and any(path.split("./")[1].startswith(x) for x in NONCORE_DIRS): continue
    for name in files:
      if not name.endswith(".py"): continue
      if 'tinygrad/runtime/autogen' in path.replace('\\', '/'): continue
      fullpath = os.path.join(path, name)
      code = pathlib.Path(fullpath).read_text()
      ret.append((fullpath.split("tinygrad/", 1)[1], code))
  return ret

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Analyze and optionally save tinygrad code.")
  parser.add_argument("--output", help="Output file to write the combined code to.")
  args = parser.parse_args()

  ret = read_code(".")

  table = []
  for name,code in ret:
    table.append([name, len(tokenizer.encode(code))])
  print(tabulate([["name", "llm tokens"]]+sorted(table, key=lambda x: -x[1]), headers="firstrow"))

  banner = "#"*40
  code_str = ''.join([f"{banner}\n# {name}\n{banner}\n\n{code}\n" for name,code in ret])
  print(f"code has {len(code_str)} chars")
  newline_count = code_str.count('\n')
  print(f"code has {newline_count} newlines")

  encoded = tokenizer.encode(code_str)
  print(f"code has {len(encoded)} tokens")

  if args.output:
    with open(args.output, 'w') as f: f.write(code_str)
    print(f"Combined code written to {args.output}")
