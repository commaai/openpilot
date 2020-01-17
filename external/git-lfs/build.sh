#!/bin/sh
wget https://github.com/github/git-lfs/releases/download/v1.4.2/git-lfs-linux-amd64-1.4.2.tar.gz
tar -xvf ./git-lfs-linux-amd64-1.4.2.tar.gz
cp ./git-lfs-1.4.2/git-lfs .

rm ./git-lfs-linux-amd64-1.4.2.tar.gz
rm -r ./git-lfs-1.4.2
