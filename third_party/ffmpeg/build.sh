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
make clean || true
./configure \
  --prefix="$X264_DIR" \
  --enable-static \
  --enable-pic \
  --disable-cli \
  --disable-opencl
make -j$(nproc)
make install
cd ..

make clean || true

# configure ffmpeg with only the components openpilot needs
#
# C++ code (loggerd, replay, cabana) needs:
#   encoders: h264, ffvhuff, aac, libx264
#   decoders: hevc, h264, ffvhuff, aac
#   muxers: mpegts, matroska, mp4, hevc
#   demuxers: hevc, matroska, mpegts, mov
#   parsers: h264, hevc, aac
#   protocols: file, pipe
#
# ffmpeg/ffprobe binaries additionally need:
#   decoders: png, rawvideo, mp4
#   encoders: png, rawvideo
#   muxers: rawvideo, image2, mp4, null
#   demuxers: rawvideo, image2, mp4
#   filters: blend, vflip, format, scale, aformat, anull
#   bsfs: extract_extradata, aac_adtstoasc
#
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
  --disable-avdevice \
  --disable-postproc \
  --disable-everything \
  --enable-encoder=libx264,h264_v4l2m2m,aac,ffvhuff,rawvideo,png \
  --enable-decoder=h264,hevc,ffvhuff,aac,rawvideo,png,mjpeg,mp3,pcm_s16le \
  --enable-muxer=mpegts,matroska,mp4,hevc,rawvideo,image2,null,mov \
  --enable-demuxer=hevc,matroska,mpegts,mov,rawvideo,image2,aac,concat \
  --enable-parser=h264,hevc,aac,mpegaudio \
  --enable-protocol=file,pipe \
  --enable-filter=blend,vflip,format,scale,aformat,anull,aresample,null \
  --enable-bsf=extract_extradata,h264_mp4toannexb,hevc_mp4toannexb,aac_adtstoasc,null \
  --enable-swresample \
  --enable-swscale \
  --extra-cflags="-I$X264_DIR/include" \
  --extra-ldflags="-L$X264_DIR/lib"

make -j$(nproc)
make install

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR/lib
mkdir -p $INSTALL_DIR/bin

rm -rf $DIR/include
mkdir -p $DIR/include
for hdr in libavcodec libavformat libavutil; do
  cp -r $DIR/ffmpeg_build/include/$hdr $DIR/include/
done

# copy static libraries and strip debug symbols
for lib in libavcodec.a libavformat.a libavutil.a libswresample.a; do
  cp $DIR/ffmpeg_build/lib/$lib $INSTALL_DIR/lib/
  strip --strip-debug $INSTALL_DIR/lib/$lib
done
cp $X264_DIR/lib/libx264.a $INSTALL_DIR/lib/
strip --strip-debug $INSTALL_DIR/lib/libx264.a

# copy binaries
cp $DIR/ffmpeg_build/bin/ffmpeg $INSTALL_DIR/bin/
cp $DIR/ffmpeg_build/bin/ffprobe $INSTALL_DIR/bin/
