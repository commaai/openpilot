#!/usr/bin/bash
set -e

cd "$(dirname "$0")"
./build.py && ./manager.py
