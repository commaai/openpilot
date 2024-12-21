#!/usr/bin/env bash

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# Take parameters as arguments
SOURCE_DIR=$1
OUTPUT_DIR=$2
DEV_BRANCH=$3
VERSION=$4
GIT_ORIGIN=$5
EXTRA_VERSION_IDENTIFIER=$6

# Check parameters
if [ -z "$SOURCE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: No source or output directory provided."
    exit 1
fi

if [ -z "$DEV_BRANCH" ] || [ -z "$VERSION" ]; then
    echo "Error: No dev branch or version provided."
    exit 1
fi

if [ -z "$GIT_ORIGIN" ]; then
    echo "Error: No GIT_ORIGIN provided"
    exit 1
fi

# "Tagging"
echo "#define COMMA_VERSION \"$VERSION\"" > ${OUTPUT_DIR}/common/version.h

## set git identity
#source $DIR/identity.sh
#export GIT_SSH_COMMAND="ssh -i /data/gitkey"

echo "[-] Setting up repo T=$SECONDS"
cd $OUTPUT_DIR
git init

# set git username/password
#source /data/identity.sh

git rm -rf $OUTPUT_DIR/.git || true # Doing cleanup, but it might fail if the .git doesn't exist or not allowed to delete
git remote remove origin || true # ensure cleanup
git remote add origin $GIT_ORIGIN
#git push origin -d $DEV_BRANCH || true # Ensuring we delete the remote branch if it exists as we are wiping it out
git fetch origin $DEV_BRANCH || (git checkout -b $DEV_BRANCH && git commit --allow-empty -m "sunnypilot v$VERSION release" && git push -u origin $DEV_BRANCH)

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git commit -a -m "sunnypilot v$VERSION release"
git branch --set-upstream-to=origin/$DEV_BRANCH

# include source commit hash and build date in commit
GIT_HASH=$(git --git-dir=$SOURCE_DIR/.git rev-parse HEAD)
DATETIME=$(date '+%Y-%m-%dT%H:%M:%S')
SP_VERSION=$(cat $SOURCE_DIR/common/version.h | awk -F\" '{print $2}')

# Add built files to git
git add -f .
if [ "$EXTRA_VERSION_IDENTIFIER" = "-release" ] || [ "$EXTRA_VERSION_IDENTIFIER" = "-staging" ]; then
  export VERSION=${VERSION%"$EXTRA_VERSION_IDENTIFIER"}
  git commit --amend -m "sunnypilot v$VERSION"
else
  git commit --amend -m "sunnypilot v$VERSION
  version: sunnypilot v$SP_VERSION release
  date: $DATETIME
  master commit: $GIT_HASH
  "
fi
git branch -m $DEV_BRANCH

# Push!
echo "[-] pushing T=$SECONDS"
git push -f origin $DEV_BRANCH
