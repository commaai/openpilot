#!/usr/bin/env bash
# usage: tools/ycpe/data/video/build.sh  (with emsdk on PATH) builds hevc.wasm next to it, FFmpeg's HEVC decoder alone,
# the software decoder full-res video falls back to without hardware HEVC or the local daemon's transcode
set -euo pipefail
cd "$(dirname "$0")"
command -v emcc >/dev/null || { echo "usage: $0 (needs emcc: source ~/emsdk/emsdk_env.sh)" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -sL https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz -o "$TMP/src.txz"
echo "7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3  $TMP/src.txz" | sha256sum -c --quiet
tar xJf "$TMP/src.txz" -C "$TMP"
F="$TMP/ffmpeg-8.1.3"

# --arch=wasm picks up FFmpeg's simd128 HEVC idct and SAO kernels
(cd "$F" && ./configure --cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm \
  --target-os=none --arch=wasm --enable-cross-compile --extra-cflags="-msimd128 -flto" \
  --disable-everything --disable-autodetect --disable-programs --disable-doc --disable-network \
  --disable-avformat --disable-avdevice --disable-avfilter --disable-swscale --disable-swresample \
  --disable-pthreads --disable-inline-asm --disable-x86asm --disable-debug \
  --enable-decoder=hevc >/dev/null)
make -C "$F" -j"$(nproc)" libavcodec/libavcodec.a libavutil/libavutil.a >/dev/null

emcc -O3 -msimd128 -flto -DNDEBUG -I"$F" hevc.c "$F"/libavcodec/libavcodec.a "$F"/libavutil/libavutil.a \
  --no-entry -sSTANDALONE_WASM -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=1MB -sINITIAL_MEMORY=32MB -sMAXIMUM_MEMORY=4GB \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_hevc_init,_hevc_push,_hevc_reset,_hevc_frame,_hevc_width,_hevc_height,_hevc_plane,_hevc_stride,_hevc_pts,_hevc_release \
  -o hevc.wasm
ls -l hevc.wasm
