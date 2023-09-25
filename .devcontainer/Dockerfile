FROM ghcr.io/commaai/openpilot-base:latest

RUN apt update && apt install -y vim net-tools usbutils htop ripgrep tmux
RUN pip install ipython jupyter jupyterlab

RUN cd $HOME && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.tmux.conf && \
    curl -O https://github.com/commaai/agnos-builder/blob/master/userspace/home/.vimrc

ENV CARLA_HOST="host.docker.internal"