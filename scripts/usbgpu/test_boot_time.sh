#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../../tinygrad_repo

sudo -E PYTHONPATH=. DEBUG=2 AM_RESET=1 AMD=1 AMD_IFACE=USB AMD_LLVM=1 time python3 test/test_tiny.py TestTiny.test_plus
