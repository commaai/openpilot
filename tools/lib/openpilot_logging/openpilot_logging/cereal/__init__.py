import os
import capnp

capnp.remove_import_hook()

try:
  # Avoids duplicate ID errors in openpilot
  from cereal import log, car, custom
except:
  # This is hit in opendbc when cereal isn't available
  CEREAL_PATH = os.path.dirname(os.path.abspath(__file__))

  log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
  car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
  custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))
