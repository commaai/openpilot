#!/usr/bin/env bash

curl -LO https://get.perfetto.dev/trace_processor
chmod +x ./trace_processor

./trace_processor --httpd
