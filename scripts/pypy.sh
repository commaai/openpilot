#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

cd ../
uv python install pypy@3.11
uv run --python pypy@3.11
