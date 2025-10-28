import ctypes, ctypes.util, os, sys, subprocess
from tinygrad.helpers import DEBUG, OSX, getenv

if sys.platform == 'win32':
  # Windows llvm distribution doesn't seem to add itself to PATH or anywhere else where it can be easily retrieved from.
  # winget also doesn't have something like `brew --prefix llvm` so just hardcode default installation path with an option to override
  LLVM_PATH = getenv('LLVM_PATH', 'C:\\Program Files\\LLVM\\bin\\LLVM-C.dll')
  if not os.path.exists(LLVM_PATH):
    raise FileNotFoundError('LLVM not found, you can install it with `winget install LLVM.LLVM` or point at a custom dll with LLVM_PATH')
elif OSX:
  # Will raise FileNotFoundError if brew is not installed
  # `brew --prefix` will return even if formula is not installed
  if not os.path.exists(brew_prefix:=subprocess.check_output(['brew', '--prefix', 'llvm@20']).decode().strip()):
    raise FileNotFoundError('LLVM not found, you can install it with `brew install llvm@20`')
  LLVM_PATH: str|None = os.path.join(brew_prefix, 'lib', 'libLLVM.dylib')
else:
  LLVM_PATH = ctypes.util.find_library('LLVM')
  # use newer LLVM if possible
  for ver in reversed(range(14, 20+1)):
    if LLVM_PATH is not None: break
    LLVM_PATH = ctypes.util.find_library(f'LLVM-{ver}')
  if LLVM_PATH is None:
    raise FileNotFoundError("No LLVM library found on the system. Install it via your distro's package manager and ensure it's findable as 'LLVM'")

if DEBUG>=3: print(f'Using LLVM at {repr(LLVM_PATH)}')
