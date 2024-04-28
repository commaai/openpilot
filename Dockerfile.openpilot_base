FROM ubuntu:20.04

ENV PYTHONUNBUFFERED 1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo tzdata locales ssh pulseaudio xvfb x11-xserver-utils gnome-screenshot && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY tools/install_ubuntu_dependencies.sh /tmp/tools/
RUN cd /tmp && \
    tools/install_ubuntu_dependencies.sh && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    # remove unused architectures from gcc for panda
    cd /usr/lib/gcc/arm-none-eabi/9.2.1 && \
    rm -rf arm/ && \
    rm -rf thumb/nofp thumb/v6* thumb/v8* thumb/v7+fp thumb/v7-r+fp.sp

# Add OpenCL
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    alien \
    unzip \
    tar \
    curl \
    xz-utils \
    dbus \
    gcc-arm-none-eabi \
    tmux \
    vim \
    lsb-core \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

ARG INTEL_DRIVER=l_opencl_p_18.1.0.015.tgz
ARG INTEL_DRIVER_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532
RUN mkdir -p /tmp/opencl-driver-intel

RUN cd /tmp/opencl-driver-intel && \
    echo INTEL_DRIVER is $INTEL_DRIVER && \
    curl -O $INTEL_DRIVER_URL/$INTEL_DRIVER && \
    tar -xzf $INTEL_DRIVER && \
    for i in $(basename $INTEL_DRIVER .tgz)/rpm/*.rpm; do alien --to-deb $i; done && \
    dpkg -i *.deb && \
    rm -rf $INTEL_DRIVER $(basename $INTEL_DRIVER .tgz) *.deb && \
    mkdir -p /etc/OpenCL/vendors && \
    echo /opt/intel/opencl_compilers_and_libraries_18.1.0.015/linux/compiler/lib/intel64_lin/libintelocl.so > /etc/OpenCL/vendors/intel.icd && \
    cd / && \
    rm -rf /tmp/opencl-driver-intel

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QTWEBENGINE_DISABLE_SANDBOX 1

RUN dbus-uuidgen > /etc/machine-id

ARG USER=batman
ARG USER_UID=1000
RUN useradd -m -s /bin/bash -u $USER_UID $USER
RUN usermod -aG sudo $USER
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USER

ENV POETRY_VIRTUALENVS_CREATE=false
ENV PYENV_VERSION=3.11.4
ENV PYENV_ROOT="/home/$USER/pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

COPY --chown=$USER pyproject.toml poetry.lock .python-version /tmp/
COPY --chown=$USER tools/install_python_dependencies.sh /tmp/tools/

RUN cd /tmp && \
    tools/install_python_dependencies.sh && \
    rm -rf /tmp/* && \
    rm -rf /home/$USER/.cache && \
    find /home/$USER/pyenv -type d -name ".git" | xargs rm -rf && \
    rm -rf /home/$USER/pyenv/versions/3.11.4/lib/python3.11/test

USER root
RUN sudo git config --global --add safe.directory /tmp/openpilot