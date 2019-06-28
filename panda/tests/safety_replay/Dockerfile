FROM ubuntu:16.04

RUN apt-get update && apt-get install -y make clang python python-pip git libarchive-dev libusb-1.0-0

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
RUN git checkout b6461274d684915f39dc45efc5292ea890698da9
