#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# n = number of pandas tested
PARALLEL=1 pytest --durations=0 *.py -n 5 --dist loadgroup -x
