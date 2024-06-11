#!/usr/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

PYTHONUNBUFFERED=1 NO_COLOR=1 CLAIM=1 PORT=4 ./debug_console.py
