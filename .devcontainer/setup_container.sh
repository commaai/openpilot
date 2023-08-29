#!/bin/bash

# Link root to new batman user
rm /home/batman/.bashrc
ln -s /root/.pyenv      /home/batman/.pyenv
ln -s /root/.pyenvrc    /home/batman/.pyenvrc
ln -s /root/.bashrc     /home/batman/.bashrc
ln -s /root/.Xauthority /home/batman/.Xauthority

# Setup permissions
sudo chown batman: /root
sudo chmod u+w /root
sudo chown batman: /tmp/scons_cache
sudo chmod u+w /tmp/scons_cache

pre-commit install