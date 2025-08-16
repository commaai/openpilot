#!/usr/bin/env bash
set -e

# To build sim and docs, you can run the following to mount the scons cache to the same place as in CI:
# mkdir -p .ci_cache/scons_cache
# sudo mount --bind /tmp/scons_cache/ .ci_cache/scons_cache

SCRIPT_DIR=$(dirname "$0")
OPENPILOT_DIR=$SCRIPT_DIR/../../
if [ -n "$TARGET_ARCHITECTURE" ]; then
  PLATFORM="linux/$TARGET_ARCHITECTURE"
  TAG_SUFFIX="-$TARGET_ARCHITECTURE"
else
  PLATFORM="linux/$(uname -m)"
  TAG_SUFFIX=""
fi

source $SCRIPT_DIR/docker_common.sh $1 "$TAG_SUFFIX"
source $SCRIPT_DIR/basher

#force_rebuild=1
#force_push=1

basher_exit_code=
if [ "$force_rebuild" != 1 ]
then
  basher_pull "/var/lib/docker" "/var/lib/docker2" "$PLATFORM" "$OPENPILOT_DIR/$DOCKER_FILE" https "$REMOTE_TAG:latest"
  basher_exit_code=$?
fi

NOTREBUILD_FLAG=1
if [ "$NOTREBUILD_FLAG" != 1 ] || [ "$force_rebuild" = 1 ] || [ "$basher_exit_code" != 0 ]
then
  {
    docker buildx rm "$BUILDER_NAME" >/dev/null 2>/dev/null || true
    docker buildx create --name mybuilder --driver docker-container --use
    docker buildx inspect --bootstrap

    IMAGE_PATH="$HOME/myimage.tar"

    # Zstandard uploading is broken in docker buildx! Therefore we build it this way, and use our hooks for the upload.
    docker buildx build \
      --builder mybuilder \
      --platform $PLATFORM \
      --output type=docker,dest="$IMAGE_PATH",compression=zstd,force-recompress=true \
      --progress=plain \
      -f $OPENPILOT_DIR/$DOCKER_FILE \
      $OPENPILOT_DIR

    basher_pull "/var/lib/docker" "/var/lib/docker2" "$PLATFORM" "" file "$REMOTE_TAG:latest" "$IMAGE_PATH" &
    file_pull_pid=$!

    if [ -n "$PUSH_IMAGE" ] || [ "$force_push" = 1 ] || [ "$basher_exit_code" != 0 ]
    then
      basher_push "$IMAGE_PATH" "$REMOTE_TAG:latest"
    else
      echo "not pushing"
    fi
  } ||
  # in case using the alternate Docker pull/build/push implementation failed, fall-back to the old way of doing things:
  {
    DOCKER_BUILDKIT=1 docker buildx build --provenance false --pull --platform $PLATFORM --load --cache-to type=inline --cache-from type=registry,ref=$REMOTE_TAG -t $DOCKER_IMAGE:latest -t $REMOTE_TAG -t $LOCAL_TAG -f $OPENPILOT_DIR/$DOCKER_FILE $OPENPILOT_DIR

    if [ -n "$PUSH_IMAGE" ]; then
      docker push $REMOTE_TAG
      docker tag $REMOTE_TAG $REMOTE_SHA_TAG
      docker push $REMOTE_SHA_TAG
    fi
  }

  wait $file_pull_pid
  rm "$IMAGE_PATH"
fi
