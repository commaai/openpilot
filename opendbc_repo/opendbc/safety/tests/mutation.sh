#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source $DIR/../../../setup.sh
$DIR/install_mull.sh

GIT_REF="${GIT_REF:-origin/master}"
GIT_ROOT=$(git rev-parse --show-toplevel)
MULL_OPS="mutators: [cxx_increment, cxx_decrement, cxx_comparison, cxx_boundary, cxx_bitwise_assignment, cxx_bitwise, cxx_arithmetic_assignment, cxx_arithmetic, cxx_remove_negation]"
echo -e "$MULL_OPS" > $GIT_ROOT/mull.yml
scons --mutation -j$(nproc) -D
echo -e "timeout: 1000000\ngitDiffRef: $GIT_REF\ngitProjectRoot: $GIT_ROOT" >> $GIT_ROOT/mull.yml

mull-runner-17 --ld-search-path /lib/x86_64-linux-gnu/ ./libsafety/libsafety.so -test-program=pytest -- -n8 --ignore-glob=misra/*
