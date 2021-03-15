#!/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
OP_ROOT="$DIR/../../"

if [ -z "$BUILD" ]; then
  docker pull ghcr.io/commaai/openpilot-base:latest
else
  docker build --cache-from ghcr.io/commaai/openpilot-base:latest -t ghcr.io/commaai/openpilot-base:latest -f $OP_ROOT/Dockerfile.openpilot_base .
fi

docker run \
       -it \
       --rm \
       --volume $OP_ROOT:/tmp/openpilot \
       --workdir /tmp/openpilot \
       --env PYTHONPATH=/tmp/openpilot \
       ghcr.io/commaai/openpilot-base:latest \
       /bin/bash
