FROM ghcr.io/commaai/openpilot-base:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends \
    vim net-tools usbutils htop ripgrep tmux wget mesa-utils xvfb libxtst6 libxv1 libglu1-mesa libegl1-mesa gdb && \
    pip install ipython jupyter jupyterlab && \
    ARCH=$(arch | sed s/aarch64/arm64/ | sed s/x86_64/amd64/) && \
    curl -L -o /tmp/virtualgl.deb "https://downloads.sourceforge.net/project/virtualgl/3.1/virtualgl_3.1_$ARCH.deb" && \
    dpkg -i /tmp/virtualgl.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER batman

RUN curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.tmux.conf && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.vimrc
