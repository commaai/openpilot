#! /bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

RUST_LOG=error $DIR/run.py $@

