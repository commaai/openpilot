set -e
echo "Installing capnp"

ONE=${HOME}/openpilot
VERSION=0.6.1
wget https://capnproto.org/capnproto-c++-${VERSION}.tar.gz
tar xvf capnproto-c++-${VERSION}.tar.gz
cd capnproto-c++-${VERSION}
CXXFLAGS="-fPIC" ./configure --prefix=${ONE}/external/capnparm
make -j9

# manually build binaries statically
g++ -std=gnu++11 -I./src -I./src -DKJ_HEADER_WARNINGS -DCAPNP_HEADER_WARNINGS -DCAPNP_INCLUDE_DIR=\"/usr/local/include\" -pthread -O2 -DNDEBUG -pthread -pthread -o .libs/capnp src/capnp/compiler/module-loader.o src/capnp/compiler/capnp.o  ./.libs/libcapnpc.a ./.libs/libcapnp.a ./.libs/libkj.a -lpthread -pthread

g++ -std=gnu++11 -I./src -I./src -DKJ_HEADER_WARNINGS -DCAPNP_HEADER_WARNINGS -DCAPNP_INCLUDE_DIR=\"/usr/local/include\" -pthread -O2 -DNDEBUG -pthread -pthread -o .libs/capnpc-c++ src/capnp/compiler/capnpc-c++.o  ./.libs/libcapnp.a ./.libs/libkj.a -lpthread -pthread

g++ -std=gnu++11 -I./src -I./src -DKJ_HEADER_WARNINGS -DCAPNP_HEADER_WARNINGS -DCAPNP_INCLUDE_DIR=\"/usr/local/include\" -pthread -O2 -DNDEBUG -pthread -pthread -o .libs/capnpc-capnp src/capnp/compiler/capnpc-capnp.o  ./.libs/libcapnp.a ./.libs/libkj.a -lpthread -pthread


make -j4 install

# --------
echo "Installing c-capnp"

git clone https://github.com/commaai/c-capnproto.git
cd c-capnproto
git submodule update --init --recursive
autoreconf -f -i -s
CFLAGS="-fPIC" ./configure --prefix=${ONE}/external/capnparm
make -j4

# manually build binaries statically
gcc -fPIC -o .libs/capnpc-c compiler/capnpc-c.o compiler/schema.capnp.o compiler/str.o  ./.libs/libcapnp_c.a -Wl,-rpath -Wl,${ONE}/external/capnp/lib

make install

# --------
echo "Installing java-capnp"

git clone https://github.com/dwrensha/capnproto-java.git
cd capnproto-java
git reset --hard 2c43bd712fb218da0eabdf241a750b9c05903e8e
g++ compiler/src/main/cpp/capnpc-java.c++ -std=c++11 -pthread -I${ONE}/external/capnp/include -L${ONE}/external/capnp/lib -l:libcapnp.a -l:libkj.a -pthread -lpthread -o capnpc-java
cp capnpc-java ${ONE}/external/capnp/bin/

rm -rf capnproto-c++-${VERSION}.tar.gz
rm -rf capnproto-c++-${VERSION}
