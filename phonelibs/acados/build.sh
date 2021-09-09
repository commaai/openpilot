#!/usr/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME="x86_64"
BLAS_TARGET="X64_AUTOMATIC"
if [ -f /TICI ]; then
  ARCHNAME="larch64"
  BLAS_TARGET="ARMV8A_ARM_CORTEX_A57"
elif [ -f /EON ]; then
  ARCHNAME="aarch64"
  BLAS_TARGET="ARMV8A_ARM_CORTEX_A57"
fi

if [ ! -d acados_repo/ ]; then
  #git clone https://github.com/acados/acados.git $DIR/acados
  git clone https://github.com/commaai/acados.git $DIR/acados_repo
fi
cd acados_repo
git fetch
git checkout 05bcbfe42818738c74572f27d06ad75a28d3b380
git submodule update --recursive --init

# build
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -UBLASFEO_TARGET -DBLASFEO_TARGET=$BLAS_TARGET ..
make -j4 install

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

rm $DIR/acados_repo/lib/*.json

cp -r $DIR/acados_repo/include $DIR
cp -r $DIR/acados_repo/lib $INSTALL_DIR
cp -r $DIR/acados_repo/interfaces/acados_template/acados_template $DIR/../../pyextra
#pip3 install -e $DIR/acados/interfaces/acados_template

# hack to workaround no rpath on android
if [ -f /EON ]; then
  pushd $INSTALL_DIR/lib
  for lib in $(ls .); do
    if ! readlink $lib; then
      patchelf --set-soname "$PWD/$lib" $lib
    fi
  done
  popd
fi

# build tera
# build with commaai/termux-packages for NEOS
if [ ! -f /EON ]; then
  cd $DIR/acados_repo/interfaces/acados_template/tera_renderer/
  cargo build --verbose --release
  cp target/release/t_renderer $INSTALL_DIR/
fi
