#!/usr/bin/bash -e

# git diff --name-status origin/release3-staging | grep "^A" | less

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

cd $DIR

BUILD_DIR=/data/openpilot
SOURCE_DIR="$(git rev-parse --show-toplevel)"

if [ -f /TICI ]; then
  FILES_SRC="release/files_tici"
  RELEASE_BRANCH=release3-staging
  DASHBAM_BRANCH=dashcam3-staging
elif [ -f /EON ]; then
  FILES_SRC="release/files_eon"
  RELEASE_BRANCH=release2-staging
  DASHBAM_BRANCH=dashcam-staging
else
  exit 0
fi

# set git identity
source $DIR/identity.sh

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
cp -pR --parents $(cat release/files_common) $BUILD_DIR/
cp -pR --parents $(cat $FILES_SRC) $BUILD_DIR/

# in the directory
cd $BUILD_DIR

rm -f panda/board/obj/panda.bin.signed

VERSION=$(cat selfdrive/common/version.h | awk -F[\"-]  '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-release\"" > selfdrive/common/version.h

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git commit -a -m "openpilot v$VERSION release"
git branch --set-upstream-to=origin/$RELEASE_BRANCH

# Build panda firmware
pushd panda/
#CERT=/data/pandaextra/certs/release RELEASE=1 scons -u .
CERT=/data/pandaextra/certs/release scons -u .
mv board/obj/panda.bin.signed /tmp/panda.bin.signed
popd

# Build
export PYTHONPATH="$BUILD_DIR"
scons -j$(nproc)

# Ensure no submodules in release
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
rm -rf panda/board panda/certs panda/crypto
rm -rf .sconsign.dblite Jenkinsfile release/
rm models/supercombo.dlc

# Move back signed panda fw
mkdir -p panda/board/obj
mv /tmp/panda.bin.signed panda/board/obj/panda.bin.signed

# Restore third_party
git checkout third_party/

# Mark as prebuilt release
touch prebuilt

# Add built files to git
git add -f .
git commit --amend -m "openpilot v$VERSION"

# Run tests
TEST_FILES="tools/"
cd $SOURCE_DIR
cp -pR -n --parents $TEST_FILES $BUILD_DIR/
cd $BUILD_DIR
RELEASE=1 selfdrive/test/test_onroad.py
#selfdrive/manager/test/test_manager.py
selfdrive/car/tests/test_car_interfaces.py
rm -rf $TEST_FILES

if [ ! -z "$PUSH" ]; then
  echo "[-] pushing T=$SECONDS"
  git push -f origin $RELEASE_BRANCH

  # Create dashcam
  git rm selfdrive/car/*/carcontroller.py
  git commit -m "create dashcam release from release"
  git push -f origin $RELEASE_BRANCH:$DASHCAM_BRANCH
fi

echo "[-] done T=$SECONDS"
