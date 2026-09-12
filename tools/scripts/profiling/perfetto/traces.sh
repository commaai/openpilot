#!/usr/bin/env bash

DEST=tici:/data/openpilot/selfdrive/debug/profiling/perfetto

scp "$DEST/trace_*" .
