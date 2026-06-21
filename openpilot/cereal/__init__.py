import os
import capnp
import sys
from importlib.resources import as_file, files

capnp.remove_import_hook()

with as_file(files("openpilot.cereal")) as fspath:
  CEREAL_PATH = fspath.as_posix()
  log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
  car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
  custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))

sys.modules.setdefault("cereal", sys.modules[__name__])
