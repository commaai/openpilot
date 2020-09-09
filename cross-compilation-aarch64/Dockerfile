FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    autoconf \
    automake \
    curl \
    ca-certificates \
    bash-completion \
    wget \
    software-properties-common \
    nano \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://raw.githubusercontent.com/docker/compose/1.26.2/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose

COPY sources.list /etc/apt/sources.list

# Workaround for server certificate expiration.
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421 &&\
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3EB9326A7BF6DFCD

RUN dpkg --add-architecture arm64 && \
    dpkg --print-foreign-architectures && \
    apt update

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /openpilot:${PYTHONPATH}

RUN mkdir -p /openpilot/cereal && \
    mkdir -p /openpilot/tools

COPY ./ubuntu_setup.sh /openpilot/tools/ubuntu_setup.sh
COPY ./install_capnp.sh /openpilot/cereal/install_capnp.sh
COPY ./install_libusb_arm.sh /openpilot/install_libusb_arm.sh
COPY ./install_ffmpeg_arm.sh /openpilot/install_ffmpeg_arm.sh
COPY ./Pipfile /openpilot/Pipfile
COPY ./Pipfile.lock /openpilot/Pipfile.lock

RUN cd /openpilot/tools && ./ubuntu_setup.sh -arch arm64

RUN rm -rf /openpilot/*

VOLUME ["/openpilot"]
