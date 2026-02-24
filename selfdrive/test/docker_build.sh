#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname "$0")
OPENPILOT_DIR=$SCRIPT_DIR/../../

DOCKER_IMAGE=openpilot
DOCKER_FILE=Dockerfile.openpilot
DOCKER_REGISTRY=ghcr.io/commaai
COMMIT_SHA=$(git rev-parse HEAD)

if [ -n "$TARGET_ARCHITECTURE" ]; then
  PLATFORM="linux/$TARGET_ARCHITECTURE"
  TAG_SUFFIX="-$TARGET_ARCHITECTURE"
else
  PLATFORM="linux/$(uname -m)"
  TAG_SUFFIX=""
fi

LOCAL_TAG=$DOCKER_IMAGE$TAG_SUFFIX
REMOTE_TAG=$DOCKER_REGISTRY/$LOCAL_TAG
REMOTE_SHA_TAG=$DOCKER_REGISTRY/$LOCAL_TAG:$COMMIT_SHA

DOCKER_BUILDKIT=1 docker buildx build --provenance false --pull --platform $PLATFORM --load -t $DOCKER_IMAGE:latest -t $REMOTE_TAG -t $LOCAL_TAG -f $OPENPILOT_DIR/$DOCKER_FILE $OPENPILOT_DIR

if [ -n "$PUSH_IMAGE" ]; then
  docker push $REMOTE_TAG
  docker tag $REMOTE_TAG $REMOTE_SHA_TAG
  docker push $REMOTE_SHA_TAG
fi
