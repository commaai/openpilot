#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../../tinygrad_repo

GREEN='\033[0;32m'
NC='\033[0m'


#export DEBUG=2
export PYTHONPATH=.
export AM_RESET=1
export AMD=1
export AMD_IFACE=USB
export AMD_LLVM=1

python3 -m unittest -q --buffer test.test_tiny.TestTiny.test_plus \
  > /tmp/test_tiny.log 2>&1 || (cat /tmp/test_tiny.log; exit 1)
printf "${GREEN}Booted in ${SECONDS}s${NC}\n"
printf "${GREEN}=============${NC}\n"

printf "\n\n"
printf "${GREEN}Transfer speeds:${NC}\n"
printf "${GREEN}================${NC}\n"
python3 test/external/external_test_usb_asm24.py TestDevCopySpeeds
