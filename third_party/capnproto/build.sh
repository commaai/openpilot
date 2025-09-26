#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

ARCHNAME="x86_64"
if [[ "${OSTYPE:-}" == darwin* ]]; then
  ARCHNAME="Darwin"
elif [[ -f /TICI ]]; then
  ARCHNAME="larch64"
else
  # Reflect host arch when not on macOS or TICI
  case "$(uname -m)" in
    aarch64) ARCHNAME="aarch64" ;;
    x86_64)  ARCHNAME="x86_64" ;;
    *)       ARCHNAME="$(uname -m)" ;;
  esac
fi

SRC="$DIR/src/"
CAPNP_TAG="v1.0.2"

if [[ ! -d "$SRC" ]]; then
  git clone https://github.com/capnproto/capnproto.git "$SRC"
fi

cd $SRC
git fetch --tags --depth=1 origin "$CAPNP_TAG"
git checkout -f "$CAPNP_TAG"
git reset --hard

INSTALL_PREFIX="$DIR/$ARCHNAME"
rm -rf $INSTALL_PREFIX
mkdir -p $INSTALL_PREFIX

CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=MinSizeRel
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
  -DBUILD_TESTING=OFF
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_MACOSX_RPATH=1
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
  -DWITH_OPENSSL=OFF
)
cd c++
mkdir -p build
cmake -S . -B build "${CMAKE_FLAGS[@]}"
cmake --build build -j8
cmake --install build --strip

cd "$DIR"

# Normalize include dir alongside other third_party deps
rm -rf $DIR/include
mv $DIR/$ARCHNAME/include/ $DIR/

# Optionally prune non-core libs and headers while keeping reflection
LIBDIR="$DIR/$ARCHNAME/lib"
# Keep: libcapnp, libkj (+ capnp tool in bin)
# Remove: RPC/JSON/WebSocket + KJ async/HTTP/TLS/GZip/Test + capnpc compiler lib
rm -f "$LIBDIR"/libcapnp-rpc* || true
rm -f "$LIBDIR"/libcapnp-websocket* || true
rm -f "$LIBDIR"/libcapnp-json* || true
rm -f "$LIBDIR"/libcapnpc* || true
rm -f "$LIBDIR"/libkj-async* || true
rm -f "$LIBDIR"/libkj-http* || true
rm -f "$LIBDIR"/libkj-tls* || true
rm -f "$LIBDIR"/libkj-gzip* || true
rm -f "$LIBDIR"/libkj-test* || true

rm -rf $INSTALL_PREFIX/lib/cmake/
rm -rf $INSTALL_PREFIX/lib/pkgconfig

# Prune headers for omitted components to avoid accidental use
rm -f "$DIR/include/capnp/compat/json.h" || true
rm -f "$DIR/include/capnp/compat/websocket-rpc.h" || true
rm -rf "$DIR/include/kj/async"* || true
rm -rf "$DIR/include/kj/compat/http"* || true
rm -rf "$DIR/include/kj/tls"* || true
rm -rf "$DIR/include/kj/gzip"* || true
rm -rf "$DIR/include/kj/test"* || true

echo "capnproto installed to: $INSTALL_PREFIX"
echo "- bin: $INSTALL_PREFIX/bin"
echo "- lib: $INSTALL_PREFIX/lib"
echo "- include mirrored at: $DIR/include"
