#!/usr/bin/env bash
# builds AGNOS's setup, reset and updater into one standalone executable for this machine
set -euo pipefail

OUT_DIR=$(realpath -m "${1:?usage: pack.sh OUTPUT_DIR}")
APPS="setup reset updater"

ROOT=$(realpath "$(dirname "$0")/../..")

# build from a clean copy, without scons outputs
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT

# openpilot's deps from uv.lock, without the submodules (tinygrad, msgq, ...)
# the apps use raylib's CPU renderer, which only needs libdrm
uv export -q --project "$ROOT" --frozen --no-default-groups --no-emit-project --no-hashes \
  | sed 's/^comma-deps-raylib==.*/comma-deps-raylib==6.0.0.1.post112/' > "$SRC/requirements.txt"
printf '%s\n' Nuitka==4.3.5 ordered-set==4.1.0 patchelf==0.17.2.4 >> "$SRC/requirements.txt"
VENV=${XDG_CACHE_HOME:-$HOME/.cache}/openpilot-pack/venv
uv venv -q --allow-existing --managed-python -p "$(cat "$ROOT/.python-version")" "$VENV"
VIRTUAL_ENV="$VENV" uv pip sync -q --link-mode=copy "$SRC/requirements.txt"
# debug symbols are most of the wheels' size. --link-mode=copy keeps uv's cache untouched
find "$VENV/lib" -name '*.so*' -type f -exec strip --strip-unneeded {} + 2> /dev/null || true
PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"  # patchelf

tar -C "$ROOT" -cf - --exclude=__pycache__ --exclude='*.so' --exclude=openpilot/cereal/gen \
  --exclude=openpilot/selfdrive/assets/training --exclude=openpilot/selfdrive/modeld/models \
  --exclude=openpilot/common/hardware/comma/updater \
  openpilot opendbc_repo/opendbc | tar -C "$SRC" -xf -
cd "$SRC"
export PYTHONPATH="$SRC:$SRC/opendbc_repo"

# raylib picks its backend at runtime. the comma one is only in arm64 wheels
RAYLIB=$("$PYTHON" -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('raylib_nb').origin))")
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
for ext in png ttf fnt json capnp mo po; do
  DATA+=("--include-package-data=openpilot:*.$ext" "--include-package-data=opendbc.car:*.$ext")
done

# pycapnp imports asyncio at runtime
"$PYTHON" -m nuitka --onefile --onefile-tempdir-spec="{CACHE_DIR}/agnos-apps" "${MAINS[@]}" \
  --lto=no --python-flag=isolated "${BACKENDS[@]}" --include-module=asyncio \
  "${DATA[@]}" \
  --output-dir="$SRC/build" --output-filename=agnos-apps

mkdir -p "$OUT_DIR"
mv "$SRC/build/agnos-apps" "$OUT_DIR/"

for app in $APPS; do
  ln -sfn agnos-apps "$OUT_DIR/$app"

  # smoke test, without the build's python
  status=0
  env -i RAYLIB_BACKEND=headless LC_ALL=C HOME="$SRC" timeout 10 "$OUT_DIR/$app" /bin/false /nonexistent > "$SRC/$app.log" 2>&1 || status=$?
  if [ "$status" -ne 124 ]; then
    cat "$SRC/$app.log"
    echo "$app exited with $status" >&2
    exit 1
  fi
done
echo "built $APPS in $OUT_DIR"
