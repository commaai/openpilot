FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    build-essential \
    ca-certificates \
    capnproto \
    clang \
    cppcheck \
    curl \
    git \
    libtool \
    make \
    libbz2-dev \
    libffi-dev \
    libcapnp-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libssl-dev \
    libsqlite3-dev \
    libzmq3-dev \
    llvm \
    ocl-icd-opencl-dev \
    opencl-headers \
    tk-dev \
    python-openssl \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.8.10
RUN pyenv global 3.8.10
RUN pyenv rehash

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir pre-commit==2.15.0 pylint==2.5.2

ENV PYTHONPATH=/project

WORKDIR /project
# TODO: Add tag to cereal
RUN git clone https://github.com/commaai/cereal.git /project/cereal && cd /project/cereal && git checkout d46f37c314bb92306207db44693b2f58c31f66b9

COPY SConstruct .
COPY ./site_scons /project/site_scons
COPY . /project/opendbc

RUN rm -rf /project/opendbc/.git
RUN scons -c && scons -j$(nproc)
