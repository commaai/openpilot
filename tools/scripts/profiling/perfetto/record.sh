#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "$DIR"

OUT=trace_
sudo ./tracebox -o $OUT --txt -c configs/scheduling.cfg
sudo chown "$USER:$USER" $OUT
