import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from openpilot.common.basedir import BASEDIR


def _compiler() -> str:
  for compiler in (os.getenv("CC"), "cc", "clang", "gcc"):
    if compiler and shutil.which(compiler):
      return compiler
  raise RuntimeError("no C compiler found for runtime header constants")


def load_c_constants(includes: list[str], names: list[str], include_dirs: list[str] | None = None) -> dict[str, int]:
  include_dirs = [] if include_dirs is None else include_dirs
  local_headers = [Path(BASEDIR) / include for include in includes if not include.startswith("<")]
  cache_key = hashlib.sha256(json.dumps({
    "includes": includes,
    "names": names,
    "include_dirs": include_dirs,
    "headers": [(str(path), path.stat().st_mtime_ns if path.exists() else None) for path in local_headers],
  }, sort_keys=True).encode()).hexdigest()
  cache_path = Path(tempfile.gettempdir()) / "openpilot_c_header_constants" / f"{cache_key}.json"
  if cache_path.exists():
    return json.loads(cache_path.read_text())

  include_lines = []
  for include in includes:
    if include.startswith("<"):
      include_lines.append(f"#include {include}")
    else:
      include_lines.append(f'#include "{include}"')

  print_lines = "\n".join(
    f'  printf("{name}=%llu\\n", (unsigned long long)({name}));'
    for name in names
  )
  source = "\n".join([
    *include_lines,
    "#include <stdio.h>",
    "int main(void) {",
    print_lines,
    "  return 0;",
    "}",
    "",
  ])

  compiler = _compiler()
  with tempfile.TemporaryDirectory() as tmp:
    source_path = Path(tmp) / "constants.c"
    output_path = Path(tmp) / "constants"
    source_path.write_text(source)
    cmd = [compiler, "-I", BASEDIR, *[f"-I{include_dir}" for include_dir in include_dirs], str(source_path), "-o", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    output = subprocess.check_output([str(output_path)], text=True)

  constants = {}
  for line in output.splitlines():
    name, value = line.split("=", 1)
    constants[name] = int(value)

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  cache_path.write_text(json.dumps(constants, sort_keys=True))
  return constants
