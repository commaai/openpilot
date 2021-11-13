#!/bin/sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

if [ -d /system ]; then
  if [ -f /TICI ]; then # QCOM2
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/data/pythonpath/third_party/snpe/larch64:$LD_LIBRARY_PATH"
  else # QCOM
    export LD_LIBRARY_PATH="/data/pythonpath/third_party/snpe/aarch64/:$LD_LIBRARY_PATH"
  fi
else
  # PC
  export LD_LIBRARY_PATH="$DIR/../../third_party/snpe/x86_64-linux-clang:$DIR/../../openpilot/third_party/snpe/x86_64:$LD_LIBRARY_PATH"
fi
exec ./_modeld
