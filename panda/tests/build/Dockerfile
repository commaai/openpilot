FROM ubuntu:16.04

RUN apt-get update && apt-get install -y gcc-arm-none-eabi libnewlib-arm-none-eabi python python-pip gcc g++ git autoconf gperf bison flex automake texinfo wget help2man gawk libtool libtool-bin ncurses-dev unzip unrar-free libexpat-dev sed bzip2 locales curl zlib1g-dev libffi-dev libssl-dev

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 2.7.12
RUN pyenv install 3.7.3
RUN pyenv global 3.7.3
RUN pyenv rehash

RUN pip install pycrypto==2.6.1

# Build esp toolchain
RUN mkdir -p /panda/boardesp
WORKDIR /panda/boardesp
RUN git clone --recursive https://github.com/pfalcon/esp-open-sdk.git
WORKDIR /panda/boardesp/esp-open-sdk
RUN git checkout 03f5e898a059451ec5f3de30e7feff30455f7ce
COPY ./boardesp/python2_make.py /panda/boardesp/esp-open-sdk
RUN python2 python2_make.py "CT_ALLOW_BUILD_AS_ROOT_SURE=1 make STANDALONE=y"

COPY . /panda

WORKDIR /panda
