#!/bin/bash -e

docker run --rm --privileged multiarch/qemu-user-static:register
docker pull docker.io/commaai/agnos:latest
docker run -v $PWD/../..:/tmp/openpilot docker.io/commaai/agnos /bin/bash -c  "\
         chown -R comma /tmp/openpilot && \
         su -l -c \" \
         source ~/.bash_profile && \
         cd /tmp/openpilot && \
         scons -j8 \
         \" -m \"comma\""
