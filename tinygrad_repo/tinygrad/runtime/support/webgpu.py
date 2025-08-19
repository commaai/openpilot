import ctypes, ctypes.util, os, subprocess, platform, sysconfig
from tinygrad.helpers import OSX

WEBGPU_PATH: str | None

if OSX:
  if not os.path.exists(brew_prefix:=subprocess.check_output(['brew', '--prefix', 'dawn']).decode().strip()):
    raise FileNotFoundError('dawn library not found. Install it with `brew tap wpmed92/dawn && brew install dawn`')
  WEBGPU_PATH = os.path.join(brew_prefix, 'lib', 'libwebgpu_dawn.dylib')
elif platform.system() == "Windows":
  if not os.path.exists(pydawn_path:=os.path.join(sysconfig.get_paths()["purelib"], "pydawn")):
    raise FileNotFoundError("dawn library not found. Install it with `pip install dawn-python`")
  WEBGPU_PATH = os.path.join(pydawn_path, "lib", "libwebgpu_dawn.dll")
else:
  if (WEBGPU_PATH:=ctypes.util.find_library('webgpu_dawn')) is None:
    raise FileNotFoundError("dawn library not found. " +
    "Install it with `sudo curl -L https://github.com/wpmed92/pydawn/releases/download/v0.3.0/" +
    f"libwebgpu_dawn_{platform.machine()}.so -o /usr/lib/libwebgpu_dawn.so`")
