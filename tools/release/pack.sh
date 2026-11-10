#!/usr/bin/env bash
# Builds the setup, reset and updater UIs that ship in AGNOS into one standalone executable
# with Nuitka. It bundles its own Python, raylib and dependencies, and runs the app it's
# invoked as: OUTPUT_DIR/openpilot-ui.dist/{setup,reset,updater} are symlinks to it.
#
#   tools/release/pack.sh [--onefile] OUTPUT_DIR
#
# --onefile packs it into the single file OUTPUT_DIR/openpilot-ui instead.
#
# Builds for this machine. Comma devices need arm64, so run it on a device or another arm64 machine.
# Needs uv and a C compiler. The output needs at least the glibc of the machine it's built on.
set -euo pipefail

ONEFILE=0
if [ "${1:-}" = --onefile ]; then ONEFILE=1; shift; fi
OUT_DIR=$(realpath -m "${1:?usage: pack.sh [--onefile] OUTPUT_DIR}")
APPS="setup reset updater"
MAX_COMPILED_MODULES=400

ROOT=$(realpath "$(dirname "$0")/../..")
PACK="$ROOT/tools/release/pack"

# a dedicated environment, so the apps only get the packages pinned in requirements.txt
VENV=${XDG_CACHE_HOME:-$HOME/.cache}/openpilot-pack/venv
uv venv -q --allow-existing --managed-python -p "$(cat "$ROOT/.python-version")" "$VENV"
VIRTUAL_ENV="$VENV" uv pip sync -q "$PACK/requirements.txt"
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
  # unpack into a persistent cache, not into /tmp on every start. Nuitka checks it on every start.
  MODE=(--onefile "--onefile-tempdir-spec={CACHE_DIR}/openpilot-ui/{PRODUCT}")
fi

MAINS=()
for app in $APPS; do
  MAINS+=("--main=openpilot/system/ui/$app.py")
done

# pycapnp imports asyncio at runtime, which Nuitka can't see. zstandard is only for building
# onefile, the apps don't use it.
"$PYTHON" -m nuitka "${MODE[@]}" "${MAINS[@]}" \
  --static-libpython=auto --lto=no --jobs="$(nproc)" --assume-yes-for-downloads --remove-output \
  --product-name=openpilot-ui --python-flag=no_site --python-flag=isolated \
  "${BACKENDS[@]}" --include-module=opendbc --include-module=asyncio \
  --nofollow-import-to=zstandard \
  --user-package-configuration-file="$PACK/package.yml" \
  "${DATA[@]}" \
  --report="$SRC/report.xml" \
  --output-dir="$SRC/build" --output-filename=openpilot-ui --output-folder-name=openpilot-ui.dist

# Nuitka compiles everything the apps can import, so keep their import graph small
n=$(grep -c '<module [^>]*kind="CompiledPython' "$SRC/report.xml" || true)
echo "$n compiled modules"
if [ "$n" -gt "$MAX_COMPILED_MODULES" ]; then
  echo "compiled $n modules (max $MAX_COMPILED_MODULES), check what the apps import" >&2
  exit 1
fi

if [ "$ONEFILE" = 1 ]; then
  DIST="$OUT_DIR"
  mkdir -p "$DIST"
  mv "$SRC/build/openpilot-ui" "$DIST/"
else
  DIST="$OUT_DIR/openpilot-ui.dist"
  rm -rf "$DIST"
  mkdir -p "$OUT_DIR"
  mv "$SRC/build/openpilot-ui.dist" "$DIST"
  # the wheels' libraries ship with debug symbols, which are most of their size
  find "$DIST" -name '*.so*' -type f -exec strip --strip-unneeded {} +
fi

for app in $APPS; do
  ln -sfn openpilot-ui "$DIST/$app"

  # the apps must start without the build's Python or environment
  status=0
  env -i RAYLIB_BACKEND=headless LC_ALL=C HOME="$SRC" timeout 10 "$DIST/$app" /bin/false /nonexistent > "$SRC/$app.log" 2>&1 || status=$?
  if [ "$status" -ne 124 ]; then
    cat "$SRC/$app.log"
    echo "$app exited with $status" >&2
    exit 1
  fi
done
echo "built $APPS in $DIST"
