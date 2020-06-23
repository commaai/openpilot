#!/usr/bin/env bash
set -e

mkdir -p /dev/shm
chmod 777 /dev/shm

# Write cpuset
echo $$ > /dev/cpuset/app/tasks
echo $PPID > /dev/cpuset/app/tasks


SOURCE_DIR=/data/openpilot_source
TARGET_DIR=/data/openpilot

ln -sf $TARGET_DIR /data/pythonpath

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"
export GIT_SSH_COMMAND="ssh -i /tmp/deploy_key"

echo "[-] Setting up repo T=$SECONDS"
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p $TARGET_DIR
  cd $TARGET_DIR
  git init
  git remote add origin git@github.com:commaai/openpilot.git
fi

echo "[-] fetching public T=$SECONDS"
cd $TARGET_DIR
git prune || true
git remote prune origin || true

echo "[-] bringing master-ci and devel in sync T=$SECONDS"
git fetch origin master-ci
git fetch origin devel

git checkout --track origin/master-ci || true
git reset --hard master-ci
git checkout master-ci
git reset --hard origin/devel
git clean -xdf

# leave .git alone
echo "[-] erasing old openpilot T=$SECONDS"
rm -rf $TARGET_DIR/* $TARGET_DIR/.gitmodules

# delete dotfiles in root
find . -maxdepth 1 -type f -delete

# reset tree and get version
cd $SOURCE_DIR
git clean -xdf
git checkout -- selfdrive/common/version.h

VERSION=$(cat selfdrive/common/version.h | awk -F\" '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-release\"" > selfdrive/common/version.h

# do the files copy
echo "[-] copying files T=$SECONDS"
cd $SOURCE_DIR
cp -pR --parents $(cat release/files_common) $TARGET_DIR/

# in the directory
cd $TARGET_DIR

rm -f panda/board/obj/panda.bin.signed

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git status
git commit -a -m "openpilot v$VERSION release"

# Run build
SCONS_CACHE=1 scons -j3

echo "[-] testing openpilot T=$SECONDS"
echo -n "0" > /data/params/d/Passive
echo -n "0.2.0" > /data/params/d/CompletedTrainingVersion
echo -n "1" > /data/params/d/HasCompletedSetup
echo -n "1" > /data/params/d/CommunityFeaturesToggle

PYTHONPATH="$TARGET_DIR:$TARGET_DIR/pyextra" nosetests -s selfdrive/test/test_openpilot.py
PYTHONPATH="$TARGET_DIR:$TARGET_DIR/pyextra" GET_CPU_USAGE=1 selfdrive/manager.py
PYTHONPATH="$TARGET_DIR:$TARGET_DIR/pyextra" selfdrive/car/tests/test_car_interfaces.py

echo "[-] testing panda build T=$SECONDS"
pushd panda/board/
make bin
popd

echo "[-] testing pedal build T=$SECONDS"
pushd panda/board/pedal
make obj/comma.bin
popd

if [ ! -z "$PUSH" ]; then
  echo "[-] Pushing to $PUSH T=$SECONDS"
  git push -f origin master-ci:$PUSH
fi

echo "[-] done pushing T=$SECONDS"

# reset version
cd $SOURCE_DIR
git checkout -- selfdrive/common/version.h

echo "[-] done T=$SECONDS"
