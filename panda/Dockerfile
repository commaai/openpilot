FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/tmp/pythonpath

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    g++ \
    gcc-arm-none-eabi libnewlib-arm-none-eabi \
    git \
    libffi-dev \
    libusb-1.0-0 \
    python3 \
    python3-dev \
    python3-pip \
 && rm -rf /var/lib/apt/lists/* && \
    apt clean && \
    cd /usr/lib/gcc/arm-none-eabi/* && \
    rm -rf arm/ && \
    rm -rf thumb/nofp thumb/v6* thumb/v8* thumb/v7+fp thumb/v7-r+fp.sp && \
    apt-get update && apt-get install -y clang-17 && \
    ln -s $(which clang-17) /usr/bin/clang

RUN apt-get update && apt-get install -y curl && \
    curl -1sLf 'https://dl.cloudsmith.io/public/mull-project/mull-stable/setup.deb.sh' | bash && \
    apt-get update && apt-get install -y mull-17

ENV CPPCHECK_DIR=/tmp/cppcheck
COPY tests/misra/install.sh /tmp/
RUN /tmp/install.sh && rm -rf $CPPCHECK_DIR/.git/
ENV SKIP_CPPCHECK_INSTALL=1

COPY setup.py __init__.py $PYTHONPATH/panda/
COPY python/__init__.py $PYTHONPATH/panda/python/
RUN pip3 install --break-system-packages --no-cache-dir $PYTHONPATH/panda/[dev]

# TODO: this should be a "pip install" or not even in this repo at all
RUN git config --global --add safe.directory $PYTHONPATH/panda
ENV OPENDBC_REF="da0a5e3d2b3984b56ebf5e25d9769f5c77807e4d"
RUN cd /tmp/ && \
    git clone --depth 1 https://github.com/commaai/opendbc opendbc_repo && \
    cd opendbc_repo && git fetch origin $OPENDBC_REF && git checkout FETCH_HEAD && rm -rf .git/ && \
    pip3 install --break-system-packages --no-cache-dir Cython numpy pycapnp  && \
    ln -s $PWD/opendbc $PYTHONPATH/opendbc && \
    scons -j8 --minimal opendbc/

# for Jenkins
COPY README.md panda.tar.* /tmp/
RUN mkdir -p /tmp/pythonpath/panda && \
    tar -xvf /tmp/panda.tar.gz -C /tmp/pythonpath/panda/ || true
