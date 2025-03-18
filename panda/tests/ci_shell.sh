#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
OP_ROOT="$DIR/../../"
PANDA_ROOT="$DIR/../"

if [ -z "$BUILD" ]; then
  docker pull docker.io/commaai/panda:latest
else
  docker build --cache-from docker.io/commaai/panda:latest -t docker.io/commaai/panda:latest -f $PANDA_ROOT/Dockerfile $PANDA_ROOT
fi

docker run \
       -it \
       --rm \
       --volume $OP_ROOT:$OP_ROOT \
       --workdir $PWD \
       --env PYTHONPATH=$OP_ROOT \
       docker.io/commaai/panda:latest \
       /bin/bash
