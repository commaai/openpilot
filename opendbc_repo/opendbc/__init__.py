import os

DBC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dbc')

# -I include path for e.g. "#include <opendbc/safety/safety.h>"
INCLUDE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

_generated_dbc_cache: dict[str, str] | None = None

def get_generated_dbcs() -> dict[str, str]:
  """Lazily generate all *_generated DBC content in memory.
  Returns {name: content} where name has no .dbc extension."""
  global _generated_dbc_cache
  if _generated_dbc_cache is None:
    from opendbc.dbc.generator.generator import generate_all
    _generated_dbc_cache = generate_all()
  return _generated_dbc_cache
