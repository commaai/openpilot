FROM ubuntu:16.04
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    bash \
    bison \
    bzip2 \
    curl \
    dfu-util \
    flex \
    g++ \
    gawk \
    gcc \
    git \
    gperf \
    help2man \
    iputils-ping \
    libexpat-dev \
    libstdc++-arm-none-eabi-newlib \
    libtool \
    libtool-bin \
    libusb-1.0-0 \
    make \
    ncurses-dev \
    network-manager \
    python-dev \
    python-serial \
    sed \
    texinfo \
    unrar-free \
    unzip \
    wget \
    build-essential \
    python-dev \
    python-pip \
    screen \
    vim \
    wget \
    wireless-tools

RUN pip install --upgrade pip==18.0

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /home/batman
ENV HOME /home/batman

ENV PYTHONPATH /tmp:$PYTHONPATH

COPY ./boardesp/get_sdk_ci.sh /tmp/panda/boardesp/

RUN useradd --system -s /sbin/nologin pandauser
RUN mkdir -p /tmp/panda/boardesp/esp-open-sdk
RUN chown pandauser /tmp/panda/boardesp/esp-open-sdk
USER pandauser
RUN cd /tmp/panda/boardesp && ./get_sdk_ci.sh
USER root

COPY ./xx/pandaextra /tmp/pandaextra

ADD ./panda.tar.gz /tmp/panda
