#!/bin/bash

#echo -e "\e[1;33m Do you want to download pre-built image from DockerHub? \e[0m"
#read -p " [Y/N] " -n 1 -r
#if [[ $REPLY =~ ^[Yy]$ ]]; then
    #docker pull openpilot-dev-env:latest
#else

    cp ../tools/ubuntu_setup.sh ./ubuntu_setup.sh
    cp ../cereal/install_capnp.sh ./install_capnp.sh
    cp ../Pipfile ./Pipfile
    cp ../Pipfile.lock ./Pipfile.lock

    docker build -t openpilot-dev-env:latest .

    rm -rf ./ubuntu_setup.sh
    rm -rf ./install_capnp.sh
    rm -rf ./Pipfile
    rm -rf ./Pipfile.lock
#fi
