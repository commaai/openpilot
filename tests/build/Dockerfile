FROM ubuntu:16.04

RUN apt-get update && apt-get install -y gcc-arm-none-eabi libnewlib-arm-none-eabi python python-pip gcc g++ git autoconf gperf bison flex automake texinfo wget help2man gawk libtool libtool-bin ncurses-dev unzip unrar-free libexpat-dev sed bzip2

RUN pip install pycrypto==2.6.1

# Build esp toolchain
RUN mkdir -p /panda/boardesp
WORKDIR /panda/boardesp
RUN git clone --recursive https://github.com/pfalcon/esp-open-sdk.git
WORKDIR /panda/boardesp/esp-open-sdk
RUN git checkout 03f5e898a059451ec5f3de30e7feff30455f7ce
RUN CT_ALLOW_BUILD_AS_ROOT_SURE=1 make STANDALONE=y

COPY . /panda

WORKDIR /panda
