#!/bin/bash -e
cd /tmp
brew install cmake ninja llvm@15 zlib glew flex bison boost zstd ncurses
if [ ! -d "gpuocelot" ]; then
    git clone --recurse-submodules https://github.com/gpuocelot/gpuocelot.git --depth 1
fi
cd gpuocelot/ocelot
git checkout b16039dc940dc6bc4ea0a98380495769ff35ed99
mkdir -p build
cd build
cmake .. -Wno-dev -G Ninja -DOCELOT_BUILD_TOOLS=OFF -DCMAKE_BUILD_ALWAYS=0 -DBUILD_TESTS_CUDA=OFF -DBISON_EXECUTABLE=/opt/homebrew/opt/bison/bin/bison
ninja
sudo cp libgpuocelot.dylib /usr/local/lib/
