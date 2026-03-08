#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source $DIR/../../../setup.sh

GIT_REF="${GIT_REF:-origin/master}"
GIT_ROOT=$(git rev-parse --show-toplevel)
cat > $GIT_ROOT/mull.yml <<EOF
mutators: [cxx_increment, cxx_decrement, cxx_comparison, cxx_boundary, cxx_bitwise_assignment, cxx_bitwise, cxx_arithmetic_assignment, cxx_arithmetic, cxx_remove_negation]
timeout: 1000000
gitDiffRef: $GIT_REF
gitProjectRoot: $GIT_ROOT
EOF

scons -j4 -D

mull-runner-18 --debug --ld-search-path /lib/x86_64-linux-gnu/ ./libsafety/libsafety.so -test-program=pytest -- -n8 --ignore-glob=misra/*