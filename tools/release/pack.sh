#!/usr/bin/env bash
# Builds the setup, reset and updater UIs that ship in AGNOS into standalone executables
# with Nuitka. They bundle their own Python, raylib and dependencies.
#
#   tools/release/pack.sh [--onefile] OUTPUT_DIR
#
# Builds for this machine. Comma devices need arm64, so run it on a device or another arm64 machine.
# Needs uv and a C++ compiler. The output needs at least the glibc of the machine it's built on.
set -euo pipefail

ONEFILE=0
if [ "${1:-}" = --onefile ]; then ONEFILE=1; shift; fi
OUT_DIR=$(realpath -m "${1:?usage: pack.sh [--onefile] OUTPUT_DIR}")
APPS="setup reset updater"
MAX_COMPILED_MODULES=300

ROOT=$(realpath "$(dirname "$0")/../..")
PACK="$ROOT/tools/release/pack"

# a dedicated environment, so the apps only get the packages pinned in requirements.txt
VENV=${XDG_CACHE_HOME:-$HOME/.cache}/openpilot-pack/venv
uv venv -q --allow-existing -p 3.12 "$VENV"
VIRTUAL_ENV="$VENV" uv pip install -q -r "$PACK/requirements.txt"
PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"  # patchelf
pkg_dir() { "$PYTHON" -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('$1').origin))"; }

# build from a copy, so the generated files and libraries don't mix with a scons build
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT
tar -C "$ROOT" -cf - --exclude=__pycache__ --exclude='*.so' --exclude=openpilot/cereal/gen \
  --exclude=openpilot/selfdrive/assets/training --exclude=openpilot/selfdrive/modeld/models \
  --exclude=openpilot/common/hardware/comma/updater \
  openpilot opendbc_repo/opendbc | tar -C "$SRC" -xf -
cd "$SRC"
export PYTHONPATH="$SRC:$SRC/opendbc_repo"

# the native half of Params, from the same sources as params_python in openpilot/common/SConscript
CAPNP=$(pkg_dir capnproto)/install
JSON11=$(pkg_dir json11)/install
mkdir -p openpilot/cereal/gen/cpp
PATH="$CAPNP/bin:$PATH" capnp compile --src-prefix=openpilot/cereal --src-prefix=opendbc_repo/opendbc/car \
  -Iopenpilot/cereal -Iopendbc_repo/opendbc/car -oc++:openpilot/cereal/gen/cpp \
  openpilot/cereal/{log,custom,deprecated}.capnp opendbc_repo/opendbc/car/car.capnp
"${CXX:-c++}" -std=c++20 -O2 -fPIC -shared -D__COMMA_HARDWARE__=1 \
  -I. -Iopenpilot -Iopenpilot/cereal/gen/cpp -I"$CAPNP/include" -I"$JSON11/include" \
  openpilot/common/{params_c,params,swaglog,util}.cc -L"$JSON11/lib" -ljson11 -pthread \
  -o openpilot/common/libparams_c.so

# raylib_nb picks its backend at runtime. Ship the comma one (only in arm64 wheels) and headless.
# The desktop one needs OpenGL and X11, which the devices don't have.
# The comma backend links libdrm, ship it alongside, see package.yml
RAYLIB=$(pkg_dir raylib_nb)
BACKENDS=(--include-module=raylib_nb._raylib_headless)
if compgen -G "$RAYLIB/_raylib_comma.*so" > /dev/null; then
  BACKENDS+=(--include-module=raylib_nb._raylib_comma)
  cp -L "$(ldconfig -p | awk '/libdrm.so.2 /{print $NF; exit}')" "$RAYLIB/"
fi

# the data files pack.py used to ship
DATA=()
while IFS= read -r f; do
  DATA+=("--include-data-files=$f=${f#opendbc_repo/}")
done < <(find openpilot opendbc_repo/opendbc/car -type f \
           \( -name '*.png' -o -name '*.ttf' -o -name '*.capnp' -o -name '*.json' -o -name '*.fnt' -o -name '*.mo' -o -name '*.po' \) \
           -not -path 'openpilot/selfdrive/assets/training/*' -not -path '*/tests/*' | sort)

MODE=(--standalone)
if [ "$ONEFILE" = 1 ]; then
  # unpack once per build into a persistent cache, not into /tmp on every start
  HASH=$(find openpilot opendbc_repo/opendbc -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
  MODE=(--onefile "--onefile-tempdir-spec={CACHE_DIR}/openpilot-ui/{PRODUCT}-$HASH")
fi

mkdir -p "$OUT_DIR"
for app in $APPS; do
  rm -rf "${OUT_DIR:?}/$app" "${OUT_DIR:?}/$app.dist"
  # pycapnp imports asyncio at runtime, which Nuitka can't see
  "$PYTHON" -m nuitka "${MODE[@]}" \
    --static-libpython=auto --lto=no --jobs="$(nproc)" --assume-yes-for-downloads \
    --product-name="$app" --python-flag=no_site --python-flag=isolated \
    "${BACKENDS[@]}" --include-module=opendbc --include-module=asyncio \
    --user-package-configuration-file="$PACK/package.yml" \
    "${DATA[@]}" \
    --report="$OUT_DIR/$app-report.xml" \
    --output-dir="$SRC/build" --output-filename="$app" \
    "openpilot/system/ui/$app.py"
  if [ "$ONEFILE" = 1 ]; then
    mv "$SRC/build/$app" "$OUT_DIR/$app"
  else
    mv "$SRC/build/$app.dist" "$OUT_DIR/$app.dist"
  fi
done

cd "$OUT_DIR"
for app in $APPS; do
  # Nuitka compiles everything the apps can import, so keep their import graph small
  n=$(grep -c '<module [^>]*kind="CompiledPython' "$app-report.xml" || true)
  echo "$app: $n compiled modules"
  if [ "$n" -gt "$MAX_COMPILED_MODULES" ]; then
    echo "$app compiles $n modules (max $MAX_COMPILED_MODULES), check what it imports" >&2
    exit 1
  fi

  # the apps must start without the build's Python or environment
  bin="$app.dist/$app"; [ "$ONEFILE" = 1 ] && bin="$app"
  status=0
  env -i RAYLIB_BACKEND=headless LC_ALL=C HOME="$SRC" timeout 10 "./$bin" /bin/false /nonexistent > "$SRC/$app.log" 2>&1 || status=$?
  if [ "$status" -ne 124 ]; then
    cat "$SRC/$app.log"
    echo "$app exited with $status" >&2
    exit 1
  fi
done
echo "built $APPS in $OUT_DIR"
