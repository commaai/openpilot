FROM ghcr.io/commaai/openpilot-base:latest

RUN apt update && \
    apt install -y vim net-tools usbutils htop ripgrep tmux wget mesa-utils xvfb libxtst6 libxv1 libglu1-mesa libegl1-mesa gdb && \
    # Add ghcli https://github.com/cli/cli/blob/trunk/docs/install_linux.md#debian-ubuntu-linux-raspberry-pi-os-apt
    mkdir -p -m 755 /etc/apt/keyrings && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \
    apt install gh -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install ipython jupyter jupyterlab

RUN cd /tmp && \
    ARCH=$(arch | sed s/aarch64/arm64/ | sed s/x86_64/amd64/) && \
    curl -L -o virtualgl.deb "https://downloads.sourceforge.net/project/virtualgl/3.1/virtualgl_3.1_$ARCH.deb" && \
    dpkg -i virtualgl.deb

USER batman

RUN cd $HOME && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.tmux.conf && \
    curl -O https://raw.githubusercontent.com/commaai/agnos-builder/master/userspace/home/.vimrc
