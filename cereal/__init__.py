import os
import capnp
import tempfile
from pathlib import Path
from importlib.resources import files

capnp.remove_import_hook()

# Extract schemas (and their includes) to a persistent temp directory so
# relative imports like "./include/c++.capnp" resolve correctly in zipapps
_tmpdir = tempfile.TemporaryDirectory(prefix="cereal_capnp_")
CEREAL_PATH = _tmpdir.name

def _extract_resource(rel_path: str) -> str:
  pkg_path = files("cereal").joinpath(rel_path)
  dest_path = Path(CEREAL_PATH) / rel_path
  dest_path.parent.mkdir(parents=True, exist_ok=True)
  # write_bytes works for both files from filesystem and zip resources
  dest_path.write_bytes(pkg_path.read_bytes())
  return dest_path.as_posix()

# Ensure include dir exists for relative imports
_extract_resource("include/c++.capnp")

# Ensure top-level schemas and their deps exist
_extract_resource("car.capnp")
_extract_resource("legacy.capnp")
_extract_resource("custom.capnp")
_extract_resource("log.capnp")

_imports = [CEREAL_PATH]
log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"), imports=_imports)
car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"), imports=_imports)
custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"), imports=_imports)
