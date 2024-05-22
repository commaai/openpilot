if [ "$1" = "base" ]; then
  export DOCKER_IMAGE=openpilot-base
  export DOCKER_FILE=Dockerfile.openpilot_base
elif [ "$1" = "sim" ]; then
  export DOCKER_IMAGE=openpilot-sim
  export DOCKER_FILE=tools/sim/Dockerfile.sim
elif [ "$1" = "prebuilt" ]; then
  export DOCKER_IMAGE=openpilot-prebuilt
  export DOCKER_FILE=Dockerfile.openpilot
else
  echo "Invalid docker build image: '$1'"
  exit 1
fi

export DOCKER_REGISTRY=ghcr.io/commaai
export COMMIT_SHA=$(git rev-parse HEAD)

TAG_SUFFIX=$2
LOCAL_TAG=$DOCKER_IMAGE$TAG_SUFFIX
REMOTE_TAG=$DOCKER_REGISTRY/$LOCAL_TAG
REMOTE_SHA_TAG=$DOCKER_REGISTRY/$LOCAL_TAG:$COMMIT_SHA
