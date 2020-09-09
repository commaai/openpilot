#!/bin/bash

docker run --rm -it -v $(pwd)/..:/openpilot --cpus 4 openpilot-dev-env:latest /openpilot/cross-compilation-aarch64/run.sh
