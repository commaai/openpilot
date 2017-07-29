import os
import capnp

CEREAL_PATH = os.path.dirname(os.path.abspath(__file__))
capnp.remove_import_hook()

if os.getenv("NEWCAPNP"):
  import tempfile
  import pyximport

  importers = pyximport.install(build_dir=os.path.join(tempfile.gettempdir(), ".pyxbld"))
  try:
    import cereal.gen.cython.log_capnp_cython as log
    import cereal.gen.cython.car_capnp_cython as car
  finally:
    pyximport.uninstall(*importers)
    del importers
else:
  log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
  car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
