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
BUILD_DIR="$DIR/build/"

mkdir -p "$BUILD_DIR"
rm -rf include/ && mkdir -p include/
rm -rf "$PREFIX" && mkdir -p "$PREFIX"

# *** build x264 ***
if [[ ! -d "$DIR/src/x264/" ]]; then
  # TODO: pin to a commit
  git clone --depth=1 --branch "stable" https://code.videolan.org/videolan/x264.git "$DIR/src/x264/"
fi
cd $DIR/src/x264
git fetch origin b35605ace3ddf7c1a5d67a2eb553f034aef41d55
git checkout -f FETCH_HEAD
./configure --prefix="$PREFIX" --enable-static --disable-opencl --enable-pic --disable-cli
make -j8
make install
cp -a "$PREFIX/include/." "$DIR/include/"

# *** build ffmpeg ***
mkdir -p "$DIR/src"
if [[ ! -d "$DIR/src/ffmpeg-$VERSION" ]]; then
  echo "Downloading FFmpeg $VERSION ..."
  curl -L "https://ffmpeg.org/releases/ffmpeg-${VERSION}.tar.xz" -o "$DIR/src/ffmpeg-${VERSION}.tar.xz"
  tar -C "$DIR/src" -xf "$DIR/src/ffmpeg-${VERSION}.tar.xz"
fi

cd $BUILD_DIR
case "$ARCHNAME" in
  x86_64)
    # Ensure vendored x264 for libx264 encoder used in encoderd on PC
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export EXTRA_CFLAGS="-I$PREFIX/include ${EXTRA_CFLAGS:-}"
    export EXTRA_LDFLAGS="-L$PREFIX/lib ${EXTRA_LDFLAGS:-}"
    # Configure minimal static FFmpeg for desktop Linux tools
    "$DIR/src/ffmpeg-$VERSION/configure" \
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

make -j$(nproc)
make install
cp -a "$PREFIX/include/." "$DIR/include/"

# *** cleanup ***
cd $PREFIX
rm -rf share/ doc/ man/ examples/ examples/ include/ lib/pkgconfig/
rm -f lib/libavfilter* "$DIR/include/libavfilter*"
