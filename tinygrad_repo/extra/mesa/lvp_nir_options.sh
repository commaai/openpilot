#!/bin/sh

if [ "$#" -ne 1 ] || ! [ -d $1 ]; then
  echo "usage: $0 MESA_PREFIX"
  exit 1
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

(
  cat <<EOF
#define HAVE_ENDIAN_H
#define HAVE_STRUCT_TIMESPEC
#define HAVE_PTHREAD
#include <unistd.h>
#include "nir_shader_compiler_options.h"
#include "compiler/shader_enums.h"
EOF
  sed -n '/struct nir_shader_compiler_options/,/^}/{p;/^}/q}' $1/src/gallium/drivers/llvmpipe/lp_screen.c
  echo "int main(void) { write(1, &gallivm_nir_options, sizeof(gallivm_nir_options)); }"
) | cc -x c -o $TMP - -I$1/src/compiler/nir -I$1/src -I$1/include || exit 1

printf 'lvp_nir_options = gzip.decompress(base64.b64decode("%s"))' $("$TMP" | gzip | base64 -w0)
