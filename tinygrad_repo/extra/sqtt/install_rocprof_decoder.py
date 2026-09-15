#!/usr/bin/env python3
import os, platform, shutil, subprocess
from pathlib import Path
from tinygrad.helpers import fetch, OSX

VERSION = "0.1.6"
DEST = Path("/usr/local/lib")
DEST.mkdir(exist_ok=True)

if __name__ == "__main__":
  if OSX:
    arch = "arm64" if platform.machine() == "arm64" else "x86_64"
    dmg = fetch(f"https://github.com/ROCm/rocprof-trace-decoder/releases/download/{VERSION}/rocprof-trace-decoder-macos-{arch}-{VERSION}-Darwin.dmg")
    mnt = Path(subprocess.check_output(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountrandom", "/tmp", str(dmg)],
                                       text=True).split("\t")[-1].strip())
    try: shutil.copy2(next(mnt.rglob("librocprof-trace-decoder.dylib")), DEST)
    finally: subprocess.run(["hdiutil", "detach", str(mnt)], check=True)
    lib = DEST/"librocprof-trace-decoder.dylib"
  else:
    lib = DEST/"librocprof-trace-decoder.so"
    os.system(f"sudo curl -L https://github.com/ROCm/rocprof-trace-decoder/raw/{VERSION}/releases/linux_glibc_2_28_x86_64/librocprof-trace-decoder.so -o {lib}")
    os.system("sudo ldconfig")
  print(f"Installed {lib.name} ({VERSION}) to", DEST)
