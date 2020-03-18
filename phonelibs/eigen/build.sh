#!/bin/bash
set -e
files=(Eigen/ unsupported/ COPYING* signature* INSTALL README.md)
rm -rf "${files[@]}"
rm -rf ./eigen-eigen* eigen.tar.gz

wget http://bitbucket.org/eigen/eigen/get/3.3.3.tar.gz -O eigen.tar.gz
tar -xzf eigen.tar.gz
mv ${files[@]/#/./eigen-eigen*/} .
find . -type d -name test | xargs rm -rf
find . -type d -name bench | xargs rm -rf

rm -rf ./eigen-eigen* eigen.tar.gz
