#!/usr/bin/bash

if [ ! -d perfetto ]; then
  git clone https://android.googlesource.com/platform/external/perfetto/
fi

cd perfetto

tools/install-build-deps --linux-arm
tools/gn gen --args='is_debug=false target_os="linux" target_cpu="arm64"' out/linux
tools/ninja -C out/linux tracebox traced traced_probes perfetto
