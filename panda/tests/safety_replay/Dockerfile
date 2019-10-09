FROM ubuntu:16.04

RUN apt-get update && apt-get install -y make clang python python-pip git libarchive-dev libusb-1.0-0 locales curl zlib1g-dev libffi-dev bzip2 libssl-dev libbz2-dev

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.7.3
RUN pyenv global 3.7.3
RUN pyenv rehash

COPY tests/safety_replay/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY tests/safety_replay/install_capnp.sh install_capnp.sh
RUN ./install_capnp.sh

RUN mkdir /openpilot
WORKDIR /openpilot
RUN git clone https://github.com/commaai/cereal.git || true
WORKDIR /openpilot/cereal
RUN git checkout f7043fde062cbfd49ec90af669901a9caba52de9
COPY . /openpilot/panda

WORKDIR /openpilot/panda/tests/safety_replay
RUN git clone https://github.com/commaai/openpilot-tools.git tools || true
WORKDIR tools
RUN git checkout d69c6bc85f221766305ec53956e9a1d3bf283160
