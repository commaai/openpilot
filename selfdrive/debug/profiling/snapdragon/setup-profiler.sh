#!/bin/bash

# install depends
sudo apt update
sudo apt-get install -y libc++1 libc++abi1 default-jre android-tools-adb libgtk-3-0

# setup mono
sudo apt install ca-certificates gnupg
sudo gpg --homedir /tmp --no-default-keyring --keyring /usr/share/keyrings/mono-official-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
echo "deb [signed-by=/usr/share/keyrings/mono-official-archive-keyring.gpg] https://download.mono-project.com/repo/ubuntu stable-focal main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
sudo apt update
sudo apt install -y mono-complete

# download  nuget
sudo wget https://dist.nuget.org/win-x86-commandline/latest/nuget.exe -O ./nuget.exe

# install depends
mono nuget.exe install gtksourcesharp -Version 3.24.24.38 -OutputDirectory ~/.local/share/SDP

echo "Setup successful, you should now be able to run the profiler with cd SnapdragonProfiler and ./run_sdp.sh"