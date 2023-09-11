#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <base|docs|sim|prebuilt|cl> <arch1> <arch2> ..."
  exit 1
fi

SCRIPT_DIR=$(dirname "$0")
ARCHS=("${@:2}")

source $SCRIPT_DIR/docker_common.sh $1

MANIFEST_AMENDS=""
for ARCH in ${ARCHS[@]}; do
  MANIFEST_AMENDS="$MANIFEST_AMENDS --amend $REMOTE_TAG-$ARCH:$COMMIT_SHA"
done

docker manifest create $REMOTE_TAG $MANIFEST_AMENDS
docker manifest create $REMOTE_SHA_TAG $MANIFEST_AMENDS

if [[ -n "$PUSH_IMAGE" ]]; then
  docker manifest push $REMOTE_TAG
  docker manifest push $REMOTE_SHA_TAG
fi
