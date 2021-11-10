#!/bin/bash
cd "$(dirname "$0")"
sudo env PATH=$PATH py-spy record -d 5 -o /tmp/perf$1.svg -p $1 &&
google-chrome /tmp/perf$1.svg
