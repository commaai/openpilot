# Use a recent Ubuntu base image.
FROM ubuntu:22.04

# Install required packages.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-tomli \
    pkg-config \
    libglib2.0-dev \
    libfdt-dev \
    libpixman-1-dev \
    zlib1g-dev \
    ninja-build \
    meson \
    wget

# Clone QEMU source (you can pin a specific version if desired)
RUN wget https://download.qemu.org/qemu-9.2.0.tar.xz && tar xvJf qemu-9.2.0.tar.xz

WORKDIR /qemu-9.2.0

RUN apt-get install -y flex bison

# Configure QEMU to build the hexagon user-mode emulator.
RUN ./configure --target-list=hexagon-linux-user && make -j$(nproc)

# Optionally, install QEMU into /usr/local (or leave it in place).
RUN make install

# delete the source (for space)
RUN cd .. && rm -rf /qemu-9.2.0

# The QEMU binaries will be in /usr/local/bin.
# Set the entrypoint to bash so you can interact with the container.
ENTRYPOINT ["/bin/bash"]