FROM ubuntu:16.04
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y build-essential clang vim screen wget bzip2 git libglib2.0-0 python-pip capnproto libcapnp-dev libzmq5-dev libffi-dev libusb-1.0-0
RUN pip install numpy==1.11.2 scipy==0.18.1 matplotlib==2.1.2

COPY requirements_openpilot.txt /tmp/
RUN pip install -r /tmp/requirements_openpilot.txt

ENV PYTHONPATH /tmp/openpilot:$PYTHONPATH

COPY ./common /tmp/openpilot/common
COPY ./cereal /tmp/openpilot/cereal
COPY ./opendbc /tmp/openpilot/opendbc
COPY ./selfdrive /tmp/openpilot/selfdrive
COPY ./phonelibs /tmp/openpilot/phonelibs
COPY ./pyextra /tmp/openpilot/pyextra

RUN mkdir -p /tmp/openpilot/selfdrive/test/out
RUN make -C /tmp/openpilot/selfdrive/controls/lib/longitudinal_mpc clean
RUN make -C /tmp/openpilot/selfdrive/controls/lib/lateral_mpc clean
