#!/usr/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME="x86_64"
if [ -f /TICI ]; then
	ARCHNAME="larch64"
elif [ -f /EON ]; then
	ARCHNAME="aarch64"
fi

if [ ! -d acados/ ]; then
	git clone https://github.com/acados/acados.git $DIR/acados
fi
cd acados
git fetch
git checkout 0334f7c7f67a52ee511037fa691552c1805493ea
git submodule update --recursive --init

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

# build
mkdir -p build
cd build
cmake -DACADOS_INSTALL_DIR=$INSTALL_DIR -DACADOS_WITH_QPOASES=ON ..
make install
rm -rf $INSTALL_DIR/cmake

# setup python
cp -r $DIR/acados/interfaces/acados_template/ $DIR/../../pyextra
