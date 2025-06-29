FROM ubuntu:24.04

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
    libbz2-dev \
    libcapnp-dev \
    libclang-rt-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libtool \
    libzmq3-dev \
    llvm \
    make \
    cmake \
    ocl-icd-opencl-dev \
    opencl-headers  \
    python3-dev \
    python3-pip \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir pyyaml Cython scons pycapnp pre-commit ruff parameterized coverage numpy pytest

WORKDIR /project/msgq
RUN cd /tmp/ && \
    git clone -b v2.x --depth 1 https://github.com/catchorg/Catch2.git && \
    cd Catch2 && \
    mv single_include/* /project/msgq/ && \
    cd .. \
    rm -rf Catch2

ENV PYTHONPATH=/project/msgq

COPY . .
RUN ls && rm -rf .git && \
    scons -c && scons -j$(nproc)
