#!/bin/bash

sudo $HOME/one/external/pyflame/pyflame -s 5 -o /tmp/perf$1.txt -p $1 &&
$HOME/one/external/pyflame/flamegraph.pl /tmp/perf$1.txt > /tmp/perf$1.svg &&
google-chrome /tmp/perf$1.svg
