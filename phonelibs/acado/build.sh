#!/usr/bin/env sh
set -e

rm -rf acado
git clone https://github.com/acado/acado.git
cd acado
git reset --hard 5adb8cbcff5a5464706a48eaf073218ac87c9044

# Clang 8 fixes
git apply ../01.patch
sed -i '100d' cmake/CompilerOptions.cmake
sed -i '100d' cmake/CompilerOptions.cmake

mkdir build
cd build
cmake -DACADO_WITH_EXAMPLES=OFF -DACADO_BUILD_STATIC=ON -DCMAKE_INSTALL_PREFIX="$HOME/openpilot/phonelibs/acado" ..
make -j$(nproc)
make install

cd ..
cd ..

rm -r x86_64
mkdir x86_64
mv lib x86_64/lib
cp acado/build/lib/* x86_64/lib/

rm -rf acado
rm -r share
