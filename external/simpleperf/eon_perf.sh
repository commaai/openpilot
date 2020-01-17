#!/bin/bash
set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"

ssh "$1" '$HOME/one/external/simpleperf/bin/android/arm64/simpleperf record --call-graph fp -a --duration 10 -o /tmp/perf.data'
scp "$1":/tmp/perf.data /tmp/perf.data
python2 report_html.py -i /tmp/perf.data -o /tmp/report.html
