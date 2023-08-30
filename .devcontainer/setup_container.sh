#!/bin/bash

USER=batman

# Link root pyenv to new batman user
unlink /home/$USER/.pyenv
unlink /home/$USER/.pyenvrc

ln -s /root/.pyenv    /home/$USER/.pyenv
ln -s /root/.pyenvrc  /home/$USER/.pyenvrc

# TODO: this takes 2-3 minutes to complete, can we do this faster somehow?
sudo chown -R batman: /root
sudo chown -R batman: /tmp/scons_cache