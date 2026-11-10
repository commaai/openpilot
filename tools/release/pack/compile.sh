#!/usr/bin/env bash
# Runs inside tools/release/pack/Dockerfile. Compiles each app into OUT_DIR/<app>.dist/,
# or into the single file OUT_DIR/<app> with ONEFILE=1.
set -euo pipefail

OUT_DIR=$(realpath -m "${1:?usage: compile.sh OUT_DIR}")
APPS="setup reset updater"
MAX_COMPILED_MODULES=300
mkdir -p "$OUT_DIR"
export PYTHONPATH="$PWD:$PWD/opendbc_repo"

# the native half of Params, from the same sources as params_python in openpilot/common/SConscript
mkdir -p openpilot/cereal/gen/cpp
capnp compile --src-prefix=openpilot/cereal --src-prefix=opendbc_repo/opendbc/car \
  -Iopenpilot/cereal -Iopendbc_repo/opendbc/car -oc++:openpilot/cereal/gen/cpp \
  openpilot/cereal/{log,custom,deprecated}.capnp opendbc_repo/opendbc/car/car.capnp
JSON11_INC=$(python3 -c 'import json11; print(json11.INCLUDE_DIR)')
JSON11_LIB=$(python3 -c 'import json11; print(json11.LIB_DIR)')
g++-10 -std=c++20 -O2 -fPIC -shared -D__COMMA_HARDWARE__=1 \
  -I. -Iopenpilot -Iopenpilot/cereal/gen/cpp -I"$JSON11_INC" \
  openpilot/common/{params_c,params,swaglog,util}.cc -L"$JSON11_LIB" -ljson11 -pthread \
  -o openpilot/common/libparams_c.so

# the data files pack.py used to ship
DATA=()
while IFS= read -r f; do
  DATA+=("--include-data-files=$f=${f#opendbc_repo/}")
done < <(find openpilot opendbc_repo/opendbc/car -type f \
           \( -name '*.png' -o -name '*.ttf' -o -name '*.capnp' -o -name '*.json' -o -name '*.fnt' -o -name '*.mo' -o -name '*.po' \) \
           -not -path 'openpilot/selfdrive/assets/training/*' -not -path '*/tests/*' | sort)

MODE=(--standalone)
if [ "${ONEFILE:-0}" = 1 ]; then
  # unpack once per build into a persistent cache, not into /tmp on every start
  HASH=$(find openpilot opendbc_repo/opendbc -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
  MODE=(--onefile "--onefile-tempdir-spec={CACHE_DIR}/openpilot-ui/{PRODUCT}-$HASH")
fi

for app in $APPS; do
  # raylib_nb picks its backend and pycapnp imports asyncio at runtime, which Nuitka can't see
  python3 -m nuitka "${MODE[@]}" \
    --static-libpython=yes --lto=no --jobs="$(nproc)" --assume-yes-for-downloads \
    --product-name="$app" --python-flag=no_site --python-flag=isolated \
    --include-module=raylib_nb._raylib_comma --include-module=raylib_nb._raylib_headless \
    --include-module=opendbc \
    --include-module=asyncio \
    --user-package-configuration-file=tools/release/pack/package.yml \
    "${DATA[@]}" \
    --report="$OUT_DIR/$app-report.xml" \
    --output-dir=/tmp/nuitka --output-filename="$app" \
    "openpilot/system/ui/$app.py"

  if [ "${ONEFILE:-0}" = 1 ]; then
    mv "/tmp/nuitka/$app" "$OUT_DIR/$app"
  else
    mv "/tmp/nuitka/$app.dist" "$OUT_DIR/$app.dist"
  fi
done

# Nuitka compiles everything the apps can import, so keep their import graph small
for app in $APPS; do
  n=$(grep -c '<module [^>]*kind="CompiledPython' "$OUT_DIR/$app-report.xml" || true)
  echo "$app: $n compiled modules"
  if [ "$n" -gt "$MAX_COMPILED_MODULES" ]; then
    echo "$app compiles $n modules (max $MAX_COMPILED_MODULES), check what it imports" >&2
    exit 1
  fi
done
