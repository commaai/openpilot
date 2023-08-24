#!/bin/bash

export DOCKER_REGISTRY=ghcr.io/commaai
export COMMIT_SHA=$(git rev-parse HEAD);

LOCAL_TAG=$DOCKER_IMAGE:latest
REMOTE_TAG=$DOCKER_REGISTRY/$LOCAL_TAG
REMOTE_SHA_TAG=$REMOTE_TAG:$COMMIT_SHA

DOCKER_BUILDKIT=1 docker build --cache-to type=inline --cache-from type=registry,ref=$REMOTE_TAG -t $REMOTE_TAG -t $LOCAL_TAG -f $DOCKER_FILE .

if [[ ! -z "$PUSH_IMAGE" ]];
then
    docker push $REMOTE_TAG
    docker tag $REMOTE_TAG $REMOTE_SHA_TAG
    docker push $REMOTE_SHA_TAG
fi