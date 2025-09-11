#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# Detect arch name like acados/build.sh
ARCHNAME="x86_64"
if [ -f /TICI ]; then
  ARCHNAME="larch64"
fi
if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

# Pinned x264 branch
BRANCH="stable"

PREFIX="$DIR/$ARCHNAME"
SRC_DIR="$DIR/src/x264-$BRANCH"

mkdir -p "$PREFIX"

case "$ARCHNAME" in
  x86_64)
    if [[ ! -d "$SRC_DIR" ]]; then
      echo "Cloning x264 ($BRANCH) ..."
      mkdir -p "$DIR/src"
      git clone --depth=1 --branch "$BRANCH" https://code.videolan.org/videolan/x264.git "$SRC_DIR"
    fi
    pushd "$SRC_DIR" >/dev/null
    ./configure       --prefix="$PREFIX"       --enable-static       --disable-opencl       --enable-pic       --disable-cli
    make -j"$(getconf _NPROCESSORS_ONLN || echo 4)"
    make install
    popd >/dev/null
    # copy headers to common include and prune extras
    mkdir -p "$DIR/include"
    cp -a "$PREFIX/include/." "$DIR/include/" 2>/dev/null || true
    rm -rf "$PREFIX/share" "$PREFIX/doc" "$PREFIX/man" "$PREFIX/examples" || true
    ;;
  larch64)
    echo "Device (larch64) build is handled separately."
    exit 1
    ;;
  Darwin)
    echo "macOS build is handled separately."
    exit 1
    ;;
  *)
    echo "Unsupported ARCHNAME: $ARCHNAME" >&2
    exit 1
    ;;
esac

echo "x264 built for $ARCHNAME at $PREFIX"
