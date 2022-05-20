#!/usr/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME="x86_64"
BLAS_TARGET="X64_AUTOMATIC"
if [ -f /TICI ]; then
  ARCHNAME="larch64"
  BLAS_TARGET="ARMV8A_ARM_CORTEX_A57"
fi

if [ ! -d acados_repo/ ]; then
  git clone https://github.com/acados/acados.git $DIR/acados_repo
  # git clone https://github.com/commaai/acados.git $DIR/acados_repo
fi
cd acados_repo
git fetch --all
git checkout 8ea8827fafb1b23b4c7da1c4cf650de1cbd73584
git submodule update --recursive --init

# build
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -UBLASFEO_TARGET -DBLASFEO_TARGET=$BLAS_TARGET ..
make -j20 install

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

rm $DIR/acados_repo/lib/*.json

rm -rf $DIR/include
cp -r $DIR/acados_repo/include $DIR
cp -r $DIR/acados_repo/lib $INSTALL_DIR
rm -rf $DIR/../../pyextra/acados_template
cp -r $DIR/acados_repo/interfaces/acados_template/acados_template $DIR/../../pyextra
#pip3 install -e $DIR/acados/interfaces/acados_template

# build tera
cd $DIR/acados_repo/interfaces/acados_template/tera_renderer/
cargo build --verbose --release
cp target/release/t_renderer $INSTALL_DIR/
