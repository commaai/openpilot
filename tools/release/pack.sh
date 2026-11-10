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

# build from a copy, so the generated files and libraries don't mix with a scons build
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT

# A dedicated environment with openpilot's own dependencies, pinned by uv.lock, without the
# submodules (tinygrad, msgq, ...). Nuitka only bundles what the apps import, and leaves out
# imports of anything not installed here, like tinygrad behind a lazy import in ui_state.
# raylib is newer than uv.lock: the apps ship the CPU renderer, which only needs libdrm.
uv export -q --project "$ROOT" --frozen --no-default-groups --no-emit-project --no-hashes \
  | sed 's/^comma-deps-raylib==.*/comma-deps-raylib==6.0.0.1.post112/' > "$SRC/requirements.txt"
printf '%s\n' Nuitka==4.3.5 ordered-set==4.1.0 patchelf==0.17.2.4 >> "$SRC/requirements.txt"
VENV=${XDG_CACHE_HOME:-$HOME/.cache}/openpilot-pack/venv
uv venv -q --allow-existing --managed-python -p "$(cat "$ROOT/.python-version")" "$VENV"
VIRTUAL_ENV="$VENV" uv pip sync -q --link-mode=copy "$SRC/requirements.txt"
# The wheels' libraries ship with debug symbols, which are most of their size. They're copies,
# not links into uv's cache, so this only touches this environment. Some aren't ELF files
# (ffmpeg ships linker scripts), strip skips those.
find "$VENV/lib" -name '*.so*' -type f -exec strip --strip-unneeded {} + 2> /dev/null || true
PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"  # patchelf

tar -C "$ROOT" -cf - --exclude=__pycache__ --exclude='*.so' --exclude=openpilot/cereal/gen \
  --exclude=openpilot/selfdrive/assets/training --exclude=openpilot/selfdrive/modeld/models \
  --exclude=openpilot/common/hardware/comma/updater \
  openpilot opendbc_repo/opendbc | tar -C "$SRC" -xf -
cd "$SRC"
export PYTHONPATH="$SRC:$SRC/opendbc_repo"

# raylib_nb picks its backend at runtime. Ship the comma one (only in arm64 wheels) and headless.
# The desktop one needs OpenGL and X11, which the devices don't have.
RAYLIB=$("$PYTHON" -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('raylib_nb').origin))")
BACKENDS=(--include-module=raylib_nb._raylib_headless)
if compgen -G "$RAYLIB/_raylib_comma.*so" > /dev/null; then
  # the comma backend links libdrm, ship it next to the backend instead of using the OS's
  LIBDRM=$(ldconfig -p | awk '/libdrm.so.2 /{print $NF; exit}')
  BACKENDS+=(--include-module=raylib_nb._raylib_comma "--include-data-files=$LIBDRM=raylib_nb/libdrm.so.2")
fi

MAINS=()
for app in $APPS; do
  MAINS+=("--main=openpilot/system/ui/$app.py")
done
DATA=()
for ext in png ttf fnt json capnp mo po; do
  DATA+=("--include-package-data=openpilot:*.$ext" "--include-package-data=opendbc.car:*.$ext")
done

# The executable unpacks into a persistent cache, not into /tmp on every start. It checks the
# cache on every start. pycapnp imports asyncio at runtime, which Nuitka can't see.
"$PYTHON" -m nuitka --onefile --onefile-tempdir-spec="{CACHE_DIR}/agnos-apps" "${MAINS[@]}" \
  --lto=no --python-flag=isolated "${BACKENDS[@]}" --include-module=asyncio \
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
