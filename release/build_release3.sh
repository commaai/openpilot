#!/usr/bin/bash -e

BUILD_DIR=/data/openpilot
SOURCE_DIR=/data/openpilot_source

ln -snf $BUILD_DIR /data/pythonpath

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"
export GIT_SSH_COMMAND="ssh -i /data/gitkey"

echo "[-] Setting up repo T=$SECONDS"
#rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
git init
git remote add origin git@github.com:commaai/openpilot.git || true

echo "[-] fetching public T=$SECONDS"
git prune || true
git remote prune origin || true

echo "[-] git fetch origin T=$SECONDS"
git fetch origin

git checkout -f -B release3-staging
git reset --hard origin/devel
git clean -xdf

# remove everything except .git
echo "[-] erasing old files T=$SECONDS"
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;

# reset tree and get version
cd $SOURCE_DIR
git clean -xdf
git checkout -- selfdrive/common/version.h

# do the files copy
echo "[-] copying files T=$SECONDS"
cd $SOURCE_DIR
cp -pR --parents $(cat release/files_common) $BUILD_DIR/
cp -pR --parents installer/continue_openpilot.sh $BUILD_DIR/

# in the directory
cd $BUILD_DIR

rm -f panda/board/obj/panda.bin.signed

VERSION=$(cat selfdrive/common/version.h | awk -F\" '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-$(git --git-dir=$SOURCE_DIR/.git rev-parse --short HEAD)-$(date '+%Y-%m-%dT%H:%M:%S')\"" > selfdrive/common/version.h

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git status
git commit -a -m "openpilot v$VERSION release"

# Build panda firmware
pushd panda/
scons
mv board/obj/panda.bin.signed /tmp/panda.bin.signed
popd

# Build
export PYTHONPATH="$BUILD_DIR"
SCONS_CACHE=1 scons -j$(nproc)

# Run tests
#python selfdrive/manager/test/test_manager.py
selfdrive/car/tests/test_car_interfaces.py

# Cleanup
find . -name '*.a' -delete
find . -name '*.o' -delete
find . -name '*.os' -delete
find . -name '*.pyc' -delete
find . -name '__pycache__' -delete
rm -rf panda/board panda/certs panda/crypto
rm -rf .sconsign.dblite Jenkinsfile release/ apk/

# Move back signed panda fw
mkdir -p panda/board/obj
mv /tmp/panda.bin.signed panda/board/obj/panda.bin.signed

# Restore phonelibs
git checkout phonelibs/

# Mark as prebuilt release
touch prebuilt

# Add built files to git
git add -f .
git commit --amend -m "openpilot v$VERSION"

if [ ! -z "$PUSH" ]; then
  git remote set-url origin git@github.com:commaai/openpilot.git

  git push -f origin release3-staging

  # Create dashcam
  #git rm selfdrive/car/*/carcontroller.py
  #git commit -m "create dashcam release from release"
  #git push -f origin release3-staging:dashcam3-staging
fi

echo "[-] done T=$SECONDS"
