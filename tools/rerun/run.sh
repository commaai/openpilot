#!/usr/bin/env bash

# TODO: remove this file once Rerun has interface to set log message level
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

RUST_LOG=warn $DIR/run.py $@

