#!/bin/bash

if [[ $DISPLAY == *xquartz* ]]; then
    echo \"export DISPLAY=host.docker.internal:0\" >> /home/batman/.bashrc
fi