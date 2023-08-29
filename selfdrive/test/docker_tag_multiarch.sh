#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <base|docs|sim|prebuilt|cl> <arch1> <arch2> ..."
  exit 1
fi

if [ $1 = "base" ]; then
  export DOCKER_IMAGE=openpilot-base
  export DOCKER_FILE=Dockerfile.openpilot_base
elif [ $1 = "docs" ]; then
  export DOCKER_IMAGE=openpilot-docs
  export DOCKER_FILE=docs/docker/Dockerfile
elif [ $1 = "sim" ]; then
  export DOCKER_IMAGE=openpilot-sim
  export DOCKER_FILE=tools/sim/Dockerfile.sim
elif [ $1 = "prebuilt" ]; then
  export DOCKER_IMAGE=openpilot-prebuilt
  export DOCKER_FILE=Dockerfile.openpilot
elif [ $1 = "cl" ]; then
  export DOCKER_IMAGE=openpilot-base-cl
  export DOCKER_FILE=Dockerfile.openpilot_base_cl
else
  echo "Invalid docker build image $1"
  exit 1
fi

export DOCKER_REGISTRY=ghcr.io/commaai
export COMMIT_SHA=$(git rev-parse HEAD)

ARCHS=("${@:2}")
REMOTE_TAG=$DOCKER_REGISTRY/$LOCAL_TAG
REMOTE_SHA_TAG=$REMOTE_TAG:$COMMIT_SHA

MANIFEST_AMENDS=""
MANIFEST_AMENDS_SHA=""
for ARCH in ${ARCHS[@]}; do
  MANIFEST_AMENDS="$MANIFEST_AMENDS --amend $REMOTE_TAG:$ARCH"
  MANIFEST_AMENDS_SHA="$MANIFEST_AMENDS_SHA --amend $REMOTE_TAG:$ARCH-$COMMIT_SHA"
done

docker manifest create $REMOTE_TAG $MANIFEST_AMENDS
docker manifest create $REMOTE_SHA_TAG $MANIFEST_AMENDS_SHA

if [[ ! -z "$PUSH_IMAGE" ]];
then
  docker manifest push $REMOTE_TAG
  docker manifest push $REMOTE_SHA_TAG
fi