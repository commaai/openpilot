#!/usr/bin/env bash
# Builds the setup, reset and updater UIs that ship in AGNOS into one self-contained executable,
# OUTPUT_DIR/agnos-apps, with Nuitka. It bundles its own Python, raylib and dependencies, and
# runs the app it's invoked as: OUTPUT_DIR/{setup,reset,updater} are symlinks to it.
#
#   tools/release/pack.sh OUTPUT_DIR
#
# Builds for this machine. Comma devices need arm64, so run it on a device or another arm64 machine.
# Needs uv and a C compiler. The output needs at least the glibc of the machine it's built on.
set -euo pipefail

OUT_DIR=$(realpath -m "${1:?usage: pack.sh OUTPUT_DIR}")
APPS="setup reset updater"

ROOT=$(realpath "$(dirname "$0")/../..")
PACK="$ROOT/tools/release/pack"

# A dedicated environment: the apps get exactly the packages in requirements.txt. Imports of
# anything else (e.g. tinygrad, behind a lazy import in ui_state) are left out of the build.
VENV=${XDG_CACHE_HOME:-$HOME/.cache}/openpilot-pack/venv
uv venv -q --allow-existing --managed-python -p "$(cat "$ROOT/.python-version")" "$VENV"
VIRTUAL_ENV="$VENV" uv pip sync -q --link-mode=copy "$PACK/requirements.txt"
# the wheels' libraries ship with debug symbols, which are most of their size. They're copies,
# not links into uv's cache, so this only touches this environment.
find "$VENV/lib" -name '*.so*' -type f -exec strip --strip-unneeded {} +
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

MAINS=()
for app in $APPS; do
  MAINS+=("--main=openpilot/system/ui/$app.py")
done

# pycapnp imports asyncio at runtime, which Nuitka can't see.
# The executable unpacks into a persistent cache, not into /tmp on every start. It checks the
# cache on every start.
"$PYTHON" -m nuitka --onefile "--onefile-tempdir-spec={CACHE_DIR}/{PRODUCT}" "${MAINS[@]}" \
  --static-libpython=auto --lto=no --jobs="$(nproc)" --assume-yes-for-downloads --remove-output \
  --product-name=agnos-apps --python-flag=no_site --python-flag=isolated \
  "${BACKENDS[@]}" --include-module=opendbc --include-module=asyncio \
  --user-package-configuration-file="$PACK/package.yml" \
  "${DATA[@]}" \
  --output-dir="$SRC/build" --output-filename=agnos-apps

mkdir -p "$OUT_DIR"
mv "$SRC/build/agnos-apps" "$OUT_DIR/"

for app in $APPS; do
  ln -sfn agnos-apps "$OUT_DIR/$app"

  # the apps must start without the build's Python or environment
  status=0
  env -i RAYLIB_BACKEND=headless LC_ALL=C HOME="$SRC" timeout 10 "$OUT_DIR/$app" /bin/false /nonexistent > "$SRC/$app.log" 2>&1 || status=$?
  if [ "$status" -ne 124 ]; then
    cat "$SRC/$app.log"
    echo "$app exited with $status" >&2
    exit 1
  fi
done
echo "built $APPS in $OUT_DIR"
