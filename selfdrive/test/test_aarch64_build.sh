#!/bin/bash

docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker pull docker.io/commaai/agnos:latest
docker run -v $PWD/../..:/tmp/openpilot docker.io/commaai/agnos /bin/bash -c \
    "su -l -c \"source ~/.bash_profile && cd /tmp/openpilot && scons -j8\" -m \"comma\""
