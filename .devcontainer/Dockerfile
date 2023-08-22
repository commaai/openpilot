FROM ghcr.io/commaai/openpilot-base:latest

# remove gitconfig if exists, since its gonna be replaced by host one
RUN rm -f /root/.gitconfig
RUN apt update && apt install -y vim net-tools usbutils htop ripgrep
RUN pip install ipython jupyter jupyterlab
