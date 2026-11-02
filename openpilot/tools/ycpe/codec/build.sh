#!/usr/bin/env bash
# rebuilds codec.wasm (zstd + bz2 decoders) from pinned upstream releases; needs emsdk on PATH
set -euo pipefail
cd "$(dirname "$0")"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fetch() {
  curl -sL "$1" -o "$TMP/src.tgz"
  echo "$2  $TMP/src.tgz" | sha256sum -c --quiet
  tar xzf "$TMP/src.tgz" -C "$TMP"
}
fetch https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
fetch https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269

Z="$TMP/zstd-1.5.7/lib"
B="$TMP/bzip2-1.0.8"
emcc -O3 -msimd128 -flto -DNDEBUG -DBZ_NO_STDIO -DZSTD_DISABLE_ASM -DZSTD_LEGACY_SUPPORT=0 \
  -I"$Z" -I"$B" codec.c \
  "$Z"/common/{entropy_common,error_private,fse_decompress,xxhash,zstd_common}.c \
  "$Z"/decompress/{huf_decompress,zstd_ddict,zstd_decompress,zstd_decompress_block}.c \
  "$B"/{bzlib,decompress,huffman,crctable,randtable,blocksort,compress}.c \
  --no-entry -sSTANDALONE_WASM -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=16MB -sMAXIMUM_MEMORY=4GB \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_zstd,_bz2,_result,_result_len,_release \
  -o codec.wasm
ls -l codec.wasm
