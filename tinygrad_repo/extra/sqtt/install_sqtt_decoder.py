#!/usr/bin/env python3
import os, shutil
from pathlib import Path
from tinygrad.helpers import fetch, OSX

DEST = Path("/usr/local/lib")
DEST.mkdir(exist_ok=True)

if __name__ == "__main__":
  if OSX:
    fp = fetch("https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.4/rocprof-trace-decoder-macos-arm64-0.1.4-Darwin.sh")
    lib = fp.parent/"rocprof-trace-decoder-macos-arm64-0.1.4-Darwin"/"lib"/"librocprof-trace-decoder.dylib"
    os.chmod(fp, 0o755)
    os.system(f"sudo {fp} --prefix={fp.parent} --include-subdir")
    shutil.copy2(lib, DEST)
  else:
    lib = DEST/"librocprof-trace-decoder.so"
    os.system("sudo curl -L https://github.com/ROCm/rocprof-trace-decoder/raw/43bf0fef74a83c3c25badfc5a09c0bd39ed8c6f9/releases/linux_glibc_2_28_x86_64/librocprof-trace-decoder.so -o"+str(lib))
    os.system("sudo ldconfig")
  print(f"Installed {lib.name} to", DEST)
