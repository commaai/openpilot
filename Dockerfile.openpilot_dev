FROM ghcr.io/commaai/openpilot-base:latest

RUN printf "\nsource /workspaces/openpilot/tools/openpilot_env.sh\n" >> ~/.bashrc
RUN apt update && apt install -y vim net-tools usbutils htop
RUN pip install ipython jupyter jupyterlab
