#!/usr/bin/bash

echo 96000 > /d/tracing/buffer_size_kb
atrace -t 10 sched workq -b 96000 > /tmp/trace.txt
