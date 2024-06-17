FROM ghcr.io/commaai/openpilot-base:latest

RUN apt update && apt install -y vim net-tools usbutils htop ripgrep tmux wget mesa-utils xvfb libxtst6 libxv1 libglu1-mesa libegl1-mesa gdb bash-completion
RUN pip install ipython jupyter jupyterlab

RUN cd /tmp && \
    ARCH=$(arch | sed s/aarch64/arm64/ | sed s/x86_64/amd64/) && \
    curl -L -o virtualgl.deb "https://downloads.sourceforge.net/project/virtualgl/3.1/virtualgl_3.1_$ARCH.deb" && \
    dpkg -i virtualgl.deb

RUN usermod -aG video batman

USER batman

RUN cd $HOME && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.tmux.conf && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.vimrc
