import importlib, pathlib
from tinygrad.helpers import system

root = (here:=pathlib.Path(__file__).parent).parents[2]

def load(name, dll, files, **kwargs):
  if not (f:=(root/(path:=kwargs.pop("path", __name__)).replace('.','/')/f"{name}.py")).exists():
    files = files() if callable(files) else files
    f.write_text(importlib.import_module("tinygrad.runtime.support.autogen").gen(dll, files, **kwargs))
  return importlib.import_module(f"{path}.{name.replace('/', '.')}")

def __getattr__(nm):
  match nm:
    case "libc": return load("libc", ["find_library('c')"], lambda: (
      [i for i in system("dpkg -L libc6-dev").split() if 'sys/mman.h' in i or 'sys/syscall.h' in i] +
      ["/usr/include/string.h", "/usr/include/elf.h", "/usr/include/unistd.h", "/usr/include/asm-generic/mman-common.h"]), use_errno=True)
    case _: raise AttributeError(f"no such autogen: {nm}")
