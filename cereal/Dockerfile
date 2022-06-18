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
    libbz2-dev \
    libcapnp-dev \
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
    ocl-icd-opencl-dev \
    opencl-headers  \
    python-openssl \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.8.10 && \
    pyenv global 3.8.10 && \
    pyenv rehash && \
    pip3 install --no-cache-dir pyyaml==5.1.2 Cython==0.29.14 scons==3.1.1 pycapnp==1.0.0 pre-commit==2.15.0 pylint==2.5.2 parameterized==0.7.4 coverage==5.1 numpy==1.21.1

WORKDIR /project/cereal/messaging
RUN git clone https://github.com/catchorg/Catch2.git && \
    cd Catch2 && \
    git checkout 229cc4823c8cbe67366da8179efc6089dd3893e9 && \
    mv single_include/catch2 ../catch2 && \
    cd .. \
    rm -rf Catch2

WORKDIR /project/cereal

ENV PYTHONPATH=/project

COPY . .
RUN rm -rf .git && \
    scons -c && scons -j$(nproc)
