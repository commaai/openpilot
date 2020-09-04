#!/bin/bash

pushd .

cd ../

echo -e "\e[1;32m Cleaning build and removing not tracked files from repository... \e[0m"
read -p "Are you sure? [Y/N] " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

scons --clean
git clean -fdx
git clean -fdX

popd

cp ./Dockerfile ../Dockerfile.cross.aarch64
cp ./sources.list ../sources.list

pushd .

cd ../

docker build -t cross-compilation -f ./Dockerfile.cross.aarch64 .

rm -rf ./Dockerfile.cross.aarch64
rm -rf ./sources.list

popd
