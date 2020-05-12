FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    autoconf \
    clang \
    curl \
    git \
    libtool \
    libssl-dev \
    libusb-1.0-0 \
    libzmq3-dev \
    locales \
    make \
    python \
    python-pip \
    wget \
    zlib1g-dev

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.7.3
RUN pyenv global 3.7.3
RUN pyenv rehash

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

WORKDIR /openpilot
RUN git clone https://github.com/commaai/opendbc.git || true
WORKDIR /openpilot/opendbc
RUN git pull && git checkout 7f3b1774dd248d4ebad91cc9de0fb1c561fab54b
WORKDIR /openpilot
RUN git clone https://github.com/commaai/cereal.git
WORKDIR /openpilot/cereal
RUN git pull && git checkout e370f79522ff7fc0b16f33f4fef420be48061206
RUN /openpilot/cereal/install_capnp.sh

RUN pip install -r /openpilot/opendbc/requirements.txt

WORKDIR /openpilot
RUN cp /openpilot/opendbc/SConstruct /openpilot
COPY . /openpilot/panda

RUN scons -c && scons -j$(nproc)
