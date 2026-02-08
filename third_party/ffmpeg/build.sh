#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME=$(uname -m)
if [ -f /TICI ]; then
  ARCHNAME="larch64"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

cd $DIR

FFMPEG_VERSION="7.1"
if [ ! -d ffmpeg_src ]; then
  git clone --branch n${FFMPEG_VERSION} --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg_src
fi

cd ffmpeg_src

# build x264 from source for libx264 support
X264_DIR="$DIR/ffmpeg_src/x264_build"
if [ ! -d x264 ]; then
  git clone --depth 1 https://code.videolan.org/videolan/x264.git
fi
cd x264
./configure \
  --prefix="$X264_DIR" \
  --enable-static \
  --enable-pic \
  --disable-cli \
  --disable-opencl
make -j$(nproc)
make install
cd ..

# configure ffmpeg with minimal features needed for openpilot
PKG_CONFIG_PATH="$X264_DIR/lib/pkgconfig" ./configure \
  --prefix="$DIR/ffmpeg_build" \
  --enable-gpl \
  --enable-libx264 \
  --enable-pic \
  --enable-static \
  --disable-shared \
  --disable-programs \
  --enable-ffmpeg \
  --enable-ffprobe \
  --disable-doc \
  --disable-htmlpages \
  --disable-manpages \
  --disable-podpages \
  --disable-txtpages \
  --disable-network \
  --disable-autodetect \
  --disable-iconv \
  --extra-cflags="-I$X264_DIR/include" \
  --extra-ldflags="-L$X264_DIR/lib"

make -j$(nproc)
make install

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR/lib
mkdir -p $INSTALL_DIR/bin

rm -rf $DIR/include
cp -r $DIR/ffmpeg_build/include $DIR

# copy static libraries and strip debug symbols
for lib in libavcodec.a libavformat.a libavutil.a libavfilter.a libswresample.a libswscale.a; do
  cp $DIR/ffmpeg_build/lib/$lib $INSTALL_DIR/lib/
  strip --strip-debug $INSTALL_DIR/lib/$lib
done
cp $X264_DIR/lib/libx264.a $INSTALL_DIR/lib/
strip --strip-debug $INSTALL_DIR/lib/libx264.a

# copy binaries
cp $DIR/ffmpeg_build/bin/ffmpeg $INSTALL_DIR/bin/
cp $DIR/ffmpeg_build/bin/ffprobe $INSTALL_DIR/bin/
