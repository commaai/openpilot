#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# Detect arch
ARCHNAME="x86_64"
if [ -f /TICI ]; then
  ARCHNAME="larch64"
fi
if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

VERSION="6.1.1"  # LTS
PREFIX="$DIR/$ARCHNAME"
SRC_DIR="$DIR/src/ffmpeg-$VERSION"
BUILD_DIR="$DIR/build/$ARCHNAME"

mkdir -p "$BUILD_DIR"
rm -rf "$PREFIX" && mkdir -p "$PREFIX"

# Fetch source
if [[ ! -d "$SRC_DIR" ]]; then
  echo "Downloading FFmpeg $VERSION ..."
  mkdir -p "$DIR/src"
  curl -L "https://ffmpeg.org/releases/ffmpeg-${VERSION}.tar.xz" -o "$DIR/src/ffmpeg-${VERSION}.tar.xz"
  tar -C "$DIR/src" -xf "$DIR/src/ffmpeg-${VERSION}.tar.xz"
fi

export SRC="$SRC_DIR"
export BUILD="$BUILD_DIR"
export PREFIX

pushd "$BUILD_DIR" >/dev/null

case "$ARCHNAME" in
  x86_64)
    # Ensure vendored x264 for libx264 encoder used in encoderd on PC
    X264_PREFIX="$DIR/../x264/x86_64"
    if [[ ! -f "$X264_PREFIX/lib/libx264.a" ]]; then
      echo "Building vendored x264 ..."
      bash "$DIR/../x264/build.sh"
    fi
    export PKG_CONFIG_PATH="$X264_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export EXTRA_CFLAGS="-I$X264_PREFIX/include ${EXTRA_CFLAGS:-}"
    export EXTRA_LDFLAGS="-L$X264_PREFIX/lib ${EXTRA_LDFLAGS:-}"
    # Configure minimal static FFmpeg for desktop Linux tools
    "$SRC/configure" \
      --prefix="$PREFIX" \
      --datadir="$PREFIX" \
      --docdir="$PREFIX" \
      --mandir="$PREFIX" \
      --enable-static --disable-shared \
      --disable-programs --disable-doc --disable-debug \
      --disable-network \
      --disable-avdevice --disable-swscale --disable-swresample --disable-postproc --disable-avfilter \
      --disable-autodetect --disable-iconv \
      --enable-avcodec --enable-avformat --enable-avutil \
      --enable-protocol=file \
      --pkg-config-flags=--static \
      --enable-gpl --enable-libx264 \
      --disable-decoders --enable-decoder=h264,hevc,aac \
      --disable-encoders --enable-encoder=libx264,ffvhuff,aac \
      --disable-demuxers --enable-demuxer=mpegts,hevc,h264,matroska,mov \
      --disable-muxers   --enable-muxer=matroska,mpegts \
      --disable-parsers  --enable-parser=h264,hevc,aac,vorbis \
      --disable-bsfs \
      --enable-small \
      --extra-cflags="${EXTRA_CFLAGS:-}" \
      --extra-ldflags="${EXTRA_LDFLAGS:-}"
    ;;
  larch64)
    echo "Device (larch64) build is handled separately."
    echo "Please provide your cross toolchain and config, then update this script."
    exit 1
    ;;
  Darwin)
    echo "macOS build is handled separately."
    echo "Please update this script with your HW accel and universal build settings."
    exit 1
    ;;
  *)
    echo "Unsupported ARCHNAME: $ARCHNAME" >&2
    exit 1
    ;;
esac

make -C "$BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN || echo 4)"
make -C "$BUILD_DIR" install
mkdir -p "$DIR/include"
cp -a "$PREFIX/include/." "$DIR/include/"
rm -rf "$PREFIX/share" "$PREFIX/doc" "$PREFIX/man" "$PREFIX/share/doc" "$PREFIX/share/man" "$PREFIX/examples" "$PREFIX/include" "$PREFIX/lib/pkgconfig"
rm -f "$PREFIX/lib/libavfilter*" "$DIR/include/libavfilter*"
