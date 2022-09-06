#!/usr/bin/bash
set -e

# git diff --name-status origin/release3-staging | grep "^A" | less

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

BUILD_DIR=/data/openpilot
SOURCE_DIR="$(git rev-parse --show-toplevel)"

FILES_SRC="release/files_tici"
RELEASE_BRANCH=release3-staging
DASHCAM_BRANCH=dashcam3-staging

# set git identity
source $DIR/identity.sh
export GIT_SSH_COMMAND="ssh -i /data/gitkey"

echo "[-] Setting up repo T=$SECONDS"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
git init
git remote add origin git@github.com:commaai/openpilot.git
git fetch origin $RELEASE_BRANCH
git checkout --orphan $RELEASE_BRANCH

# do the files copy
echo "[-] copying files T=$SECONDS"
cd $SOURCE_DIR
if [ ! -z "$(git clean -xdf -n)" ]; then
  echo "source directory is dirty. git clean -xdf and try again."
  exit 1
fi
cp -pR --parents $(cat release/files_common) $BUILD_DIR/
cp -pR --parents $(cat $FILES_SRC) $BUILD_DIR/

# in the directory
cd $BUILD_DIR

VERSION=$(cat common/version.h | awk -F[\"-]  '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-release\"" > common/version.h

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git commit -a -m "openpilot v$VERSION release"
git branch --set-upstream-to=origin/$RELEASE_BRANCH

# Build
export PYTHONPATH="$BUILD_DIR"
export CERT=/data/pandaextra/certs/release
RELEASE=1 scons -j$(nproc)

# Ensure no submodules in release branches
if test "$(git submodule--helper list | wc -l)" -gt "0"; then
  echo "submodules found:"
  git submodule--helper list
  exit 1
fi
git submodule status

# Cleanup
find . -name '*.a' -delete
find . -name '*.o' -delete
find . -name '*.os' -delete
find . -name '*.pyc' -delete
find . -name 'moc_*' -delete
find . -name '__pycache__' -delete
rm -rf panda/certs panda/crypto panda/board/bootstub*
rm -rf .sconsign.dblite Jenkinsfile
rm selfdrive/modeld/models/supercombo.onnx

# Restore third_party
git checkout third_party/

# Mark as prebuilt release
touch prebuilt

# Add built files to git
git add -f .
git commit --amend -m "openpilot v$VERSION"

# Push
if [ ! -z "$PUSH" ]; then
  echo "[-] pushing T=$SECONDS"
  git push -f origin $RELEASE_BRANCH

  # Create dashcam
  git rm selfdrive/car/*/carcontroller.py
  git commit -m "create dashcam release from release"
  git push -f origin $RELEASE_BRANCH:$DASHCAM_BRANCH
fi

echo "[-] done T=$SECONDS"
