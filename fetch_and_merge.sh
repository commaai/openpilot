#!/usr/bin/bash

cd cereal
git checkout dev
git pull
git fetch upstream
git merge upstream/master
echo "Don't forget to commit and push"
cd ..
cd opendbc
git checkout dev
git pull
git fetch upstream
git merge upstream/master
echo "Don't forget to commit and push"
cd ..
cd panda
git checkout dev
git pull
git fetch upstream
git merge upstream/master
echo "Don't forget to commit and push"
cd ..
git checkout dev
git pull
git fetch upstream
git merge upstream/master
echo "Don't forget to commit and push"
