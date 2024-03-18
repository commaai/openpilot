#!/usr/bin/bash -e

# git diff --name-status origin/release3-staging | grep "^A" | less

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

cd $DIR

SOURCE_DIR="$(git -C $DIR rev-parse --show-toplevel)"
BUILD_DIR="${BUILD_DIR:=$(mktemp -d)}"

# set git identity
source $DIR/identity.sh

echo "[-] Setting up repo T=$SECONDS"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
git init
git remote add origin git@github.com:commaai/openpilot.git
git checkout --orphan $OPENPILOT_CHANNEL

# do the files copy
echo "[-] copying files T=$SECONDS"
cd $SOURCE_DIR
release/copy_channel_files.sh $SOURCE_DIR $BUILD_DIR

cd $BUILD_DIR
VERSION=$(cat common/version.h | awk -F[\"-]  '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-release\"" > common/version.h

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git commit -a -m "openpilot v$VERSION release"

echo "[-] creating prebuilt T=$SECONDS"
release/create_prebuilt.sh $BUILD_DIR

cd $BUILD_DIR

# Ensure no submodules in release
if test "$(git submodule--helper list | wc -l)" -gt "0"; then
  echo "submodules found:"
  git submodule--helper list
  exit 1
fi
git submodule status

# Add built files to git
git add -f .
git commit --amend -m "openpilot v$VERSION"

# Run tests
TEST_FILES="tools/"
cd $SOURCE_DIR
cp -pR -n --parents $TEST_FILES $BUILD_DIR/
cd $BUILD_DIR
selfdrive/test/test_onroad.py
#selfdrive/manager/test/test_manager.py
#selfdrive/car/tests/test_car_interfaces.py
rm -rf $TEST_FILES

if [ ! -z "$OPENPILOT_CHANNEL" ]; then
  echo "[-] pushing release T=$SECONDS"
  git push -f origin $OPENPILOT_CHANNEL:$OPENPILOT_CHANNEL
fi

echo "[-] done T=$SECONDS"
