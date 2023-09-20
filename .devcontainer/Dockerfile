FROM ghcr.io/commaai/openpilot-base:latest

# remove gitconfig if exists, since its gonna be replaced by host one
RUN rm -f /root/.gitconfig

RUN apt update && apt install -y vim net-tools usbutils htop ripgrep tmux
RUN pip install ipython jupyter jupyterlab

RUN cd $HOME && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.tmux.conf && \
    curl -O https://github.com/commaai/agnos-builder/blob/master/userspace/home/.vimrc
