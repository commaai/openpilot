#!/bin/bash -e

### EXCELENT GUIDE TO STATIC COMPILATION!
### https://gist.github.com/Brainiarc7/988473b79fd5c8f0db54b92ebb47387a

cd "$(dirname "${BASH_SOURCE[0]}")"

# main binaries from ffmpeg-static
#wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
#tar xvf ffmpeg-release-64bit-static.tar.xz
#cp ffmpeg-3.2.2-64bit-static/ffmpeg bin/
#cp ffmpeg-3.2.2-64bit-static/ffprobe bin/

rm -rf nasm-2.14.02
wget http://www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.gz
tar xzvf nasm-2.14.02.tar.gz
pushd nasm-2.14.02
  ./configure
  make -j$(nproc) VERBOSE=1
  sudo make -j$(nproc) install
  make -j$(nproc) distclean
popd

rm -rf nv-codec-headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
pushd nv-codec-headers
  sudo make install
popd

rm -rf fdk-aac
git clone https://github.com/mstorsjo/fdk-aac
pushd fdk-aac
  autoreconf -fiv
  ./configure --disable-shared
  make -j$(nproc)
  sudo make -j$(nproc) install
  make -j$(nproc) distclean
popd

rm -rf x264
git clone http://git.videolan.org/git/x264.git -b stable
pushd x264/
  ./configure --enable-static --enable-pic --bit-depth=all
  make -j$(nproc) VERBOSE=1
 sudo make -j$(nproc) install VERBOSE=1
  make -j$(nproc) distclean
popd

# binary with cuda decoding enabled
#git clone --depth 1 git@github.com:commaai/ffmpeg.git
rm -rf ffmpeg
git clone https://git.ffmpeg.org/ffmpeg.git
pushd ffmpeg
  # replace npp dynamic lib refs with static lib refs
  sed 's/-lnpp\(\w\+\)/-lnpp\1_static/g' configure > configure_npp_static
  chmod ug+x configure_npp_static
  ./configure_npp_static --pkg-config-flags="--static" --enable-static --disable-debug --disable-alsa --enable-libfdk-aac --enable-libx264 --disable-ffplay --disable-libxcb --disable-sdl2 --enable-cuda-nvcc --enable-cuvid --enable-nvenc --enable-libnpp --enable-nonfree --enable-gpl --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --extra-libs="-lcudart_static -lculibos -ldl -lrt -pthread -lstdc++"
  make -j10
  # replaced by above sed and specifying --extra-libs="-lcudart_static -lculibos -ldl -lrt -pthread -lstdc++"
  ### manually build last few libs statically
  ##g++ -Llibavcodec -Llibavdevice -Llibavfilter -Llibavformat -Llibavresample -Llibavutil -Llibpostproc -Llibswscale -Llibswresample -L/usr/local/cuda/lib64  -Wl,--as-needed -Wl,-z,noexecstack -Wl,--warn-common -Wl,-rpath-link=libpostproc:libswresample:libswscale:libavfilter:libavdevice:libavformat:libavcodec:libavutil:libavresample   -o ffmpeg_g cmdutils.o ffmpeg_opt.o ffmpeg_filter.o ffmpeg.o ffmpeg_cuvid.o -lavdevice -lavfilter -lavformat -lavcodec -lswresample -lswscale -lavutil -ldl -ldl -l:libnppi_static.a -l:libnppc_static.a -l:libcudart_static.a -l:libculibos.a -lrt -lm -l:liblzma.a -l:libbz2.a -l:libz.a -pthread
  ##cp ffmpeg_g ffmpeg
  ##strip ffmpeg
popd
cp ffmpeg/ffmpeg bin/ffmpeg
