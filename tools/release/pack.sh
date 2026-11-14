#!/usr/bin/env bash
# builds AGNOS's setup, reset and updater for this machine into
#   OUTPUT_DIR/agnos-apps/: a standalone folder, for AGNOS
#   OUTPUT_DIR/updater: a single file, for openpilot to update AGNOS with
set -euo pipefail

OUT_DIR=$(realpath -m "${1:?usage: pack.sh OUTPUT_DIR}")
APPS="setup reset updater"

ROOT=$(realpath "$(dirname "$0")/../..")

# openpilot's deps from uv.lock, without the submodules (tinygrad, msgq, ...)
VENV=${XDG_CACHE_HOME:-$HOME/.cache}/openpilot-pack/venv
export UV_PROJECT_ENVIRONMENT="$VENV"
uv sync -q --project "$ROOT" --frozen --no-default-groups --no-install-project

# the apps use raylib's CPU renderer, which only needs libdrm
uv pip install -q --python "$VENV" comma-deps-raylib==6.0.0.1.post112 \
  Nuitka==4.3.5 ordered-set==4.1.0 patchelf==0.17.2.4
export PATH="$VENV/bin:$PATH"

# build from a clean copy, without scons outputs
# materialize data aliases so onefile cleanup does not encounter dangling links
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT
tar -h -C "$ROOT" -cf - --exclude=__pycache__ --exclude='*.so' --exclude=openpilot/cereal/gen \
  --exclude=openpilot/selfdrive/assets/training --exclude=openpilot/selfdrive/modeld/models \
  --exclude=openpilot/common/hardware/comma/updater \
  openpilot opendbc_repo/opendbc | tar -C "$SRC" -xf -
cd "$SRC"
export PYTHONPATH="$SRC:$SRC/opendbc_repo"

# raylib picks its backend at runtime. the comma one is only in arm64 wheels
RAYLIB=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('raylib_nb').origin))")
BACKENDS=(--include-module=raylib_nb._raylib_headless)
if compgen -G "$RAYLIB/_raylib_comma.*so" > /dev/null; then
  # ship libdrm instead of using the OS's
  LIBDRM=$(ldconfig -p | awk '/libdrm.so.2 /{print $NF; exit}')
  BACKENDS+=(--include-module=raylib_nb._raylib_comma "--include-data-files=$LIBDRM=raylib_nb/libdrm.so.2")
fi

MAINS=()
for app in $APPS; do
  MAINS+=("--main=openpilot/system/ui/$app.py")
done
DATA=()
for ext in png ttf otf fnt json capnp mo po; do
  DATA+=("--include-package-data=openpilot:*.$ext" "--include-package-data=opendbc.car:*.$ext")
done

# nuitka's zstd takes ~1GB per job, and comma devices have ~4GB
JOBS=$(awk -v cpus="$(nproc)" '/MemAvailable/ { jobs = int($2 / 1500000); print (jobs < 1 ? 1 : (jobs > cpus ? cpus : jobs)) }' /proc/meminfo)

# pycapnp's compiled extension imports asyncio, which nuitka can't see
# each launch unpacks separately in /data; AGNOS's /home and /tmp are too small
python -m nuitka --onefile '--onefile-tempdir-spec=/data/tmp/agnos-apps_{PID}_{TIME_US}_{RANDOM}' "${MAINS[@]}" \
  --jobs="$JOBS" --lto=no --python-flag=isolated "${BACKENDS[@]}" --include-module=asyncio \
  "${DATA[@]}" \
  --output-dir="$SRC/build" --output-filename=agnos-apps

DIST="$OUT_DIR/agnos-apps"
mkdir -p "$OUT_DIR"
rm -rf "$DIST"
mv "$SRC/build/setup.dist" "$DIST"
mv "$DIST/agnos-apps.bin" "$DIST/agnos-apps"
cp "$SRC/build/agnos-apps" "$OUT_DIR/updater"
# debug symbols are most of the libraries' size
find "$DIST" -name '*.so*' -type f -exec strip --strip-unneeded {} +

for app in $APPS; do
  ln -sfn agnos-apps "$DIST/$app"

  # smoke test, without the build's python
  status=0
  env -i RAYLIB_BACKEND=headless LC_ALL=C HOME="$SRC" timeout 10 "$DIST/$app" /bin/false /nonexistent > "$SRC/$app.log" 2>&1 || status=$?
  if [ "$status" -ne 124 ]; then
    cat "$SRC/$app.log"
    echo "$app exited with $status" >&2
    exit 1
  fi
done
echo "built $APPS in $DIST, and $OUT_DIR/updater"
