#!/usr/bin/env bash
cd "$(dirname "$0")"

# prereq: install emscripten: https://emscripten.org/docs/getting_started/downloads.html
EMSCRIPTEN_PATH=~/emsdk/emsdk_env.sh
source $EMSCRIPTEN_PATH
step="transformer"
initial_memory=6553600
max_memory=1500053504
exported_functions='["_net", "_malloc", "_free", "_set_buf"]'

emcc "${step}.c" \
  -O3 -msimd128 -ffast-math -flto \
  -o "${step}.js" \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s EXPORTED_FUNCTIONS="${exported_functions}" \
  -s ENVIRONMENT='worker' \
  -s FILESYSTEM=0 \
  -s EVAL_CTORS \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY="$initial_memory" \
  -s MAXIMUM_MEMORY="$max_memory"