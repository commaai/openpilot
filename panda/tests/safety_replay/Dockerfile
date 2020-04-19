FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    bzip2 \
    clang \
    curl \
    git \
    libarchive-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libffi-dev \
    libssl-dev \
    libusb-1.0-0 \
    locales \
    make \
    python \
    python-pip \
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
COPY tests/safety_replay/requirements_extra.txt requirements_extra.txt
RUN pip install -r requirements_extra.txt
COPY tests/safety_replay/install_capnp.sh install_capnp.sh
RUN ./install_capnp.sh

RUN git clone https://github.com/commaai/openpilot.git || true
WORKDIR /openpilot
RUN git pull && git checkout f9257fc75f68c673f9e433985fbe739f23310bb4
RUN git submodule update --init cereal

COPY . /openpilot/panda

WORKDIR /openpilot/panda/tests/safety_replay
