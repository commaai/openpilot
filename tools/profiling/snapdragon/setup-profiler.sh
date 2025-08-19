#!/usr/bin/env bash

# install depends
sudo apt update
sudo apt-get install libc++1 libc++abi1 default-jre android-tools-adb gtk-sharp2

# setup mono
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
sudo apt install apt-transport-https ca-certificates
echo "deb https://download.mono-project.com/repo/ubuntu stable-xenial main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
sudo apt update
sudo apt-get install -y mono-complete

echo "Setup successful, you should now be able to run the profiler with cd SnapdragonProfiler and ./run_sdp.sh"
