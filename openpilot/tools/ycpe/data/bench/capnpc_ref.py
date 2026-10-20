#!/usr/bin/env python3
"""The reference side of capnpc.mjs: schema.json's text for the .capnp files in <dir>/cereal (import path <dir>/imp),
compiled by pycapnp and walked by gen_schema.py's own add_struct(). One process per schema: capnp ids collide across
commits."""
import json
import sys
import types
from pathlib import Path

import capnp

# gen_schema.py loads the current cereal at import; it isn't needed (and its ids would clash with the ones here)
sys.modules["openpilot.cereal"] = types.ModuleType("openpilot.cereal")
sys.modules["openpilot.cereal"].log = None
from openpilot.tools.ycpe import gen_schema

if __name__ == "__main__":
  d = Path(sys.argv[1])
  capnp.remove_import_hook()
  log = capnp.load(str(d / "cereal" / "log.capnp"), imports=[str(d / "imp")])
  gen_schema.add_struct(log.Event.schema)
  sys.stdout.write(json.dumps({"structs": gen_schema.structs, "enums": gen_schema.enums}, separators=(",", ":")))
