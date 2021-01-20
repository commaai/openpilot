FROM ubuntu:20.04
ENV PYTHONUNBUFFERED 1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    bash \
    bison \
    bzip2 \
    ca-certificates \
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
    libbz2-dev \
    libexpat-dev \
    libffi-dev \
    libssl-dev \
    libstdc++-arm-none-eabi-newlib \
    libtool \
    libtool-bin \
    libusb-1.0-0 \
    locales  \
    make \
    ncurses-dev \
    network-manager \
    python-dev \
    python3-serial \
    sed \
    texinfo \
    unrar-free \
    unzip \
    wget \
    build-essential \
    python-dev \
    screen \
    vim \
    wget \
    wireless-tools \
    zlib1g-dev

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.8.2
RUN pyenv global 3.8.2
RUN pyenv rehash

RUN pip install --upgrade pip==18.0

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /home/batman
ENV HOME /home/batman

ENV PYTHONPATH /tmp:$PYTHONPATH

RUN cd /tmp && git clone https://github.com/commaai/panda_jungle.git

ADD ./panda.tar.gz /tmp/panda
