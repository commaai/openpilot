#!/usr/bin/env bash
set -e

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"

export GIT_SSH_COMMAND="ssh -i /data/gitkey"

# Create folders
rm -rf /data/openpilot
mkdir -p /data/openpilot
cd /data/openpilot

# Create git repo
git init
git remote add public git@github.com:commaai/openpilot.git
git fetch public

# Checkout devel
git checkout public/devel
git clean -xdf

# Create release2 with no history
git branch -D release2 || true
git checkout --orphan release2

VERSION=$(cat selfdrive/common/version.h | awk -F\" '{print $2}')
git commit -m "openpilot v$VERSION"

# Build signed panda firmware
pushd panda/board/
cp -r /tmp/pandaextra /data/openpilot/
RELEASE=1 make obj/panda.bin
mv obj/panda.bin /tmp/panda.bin
make clean
mv /tmp/panda.bin obj/panda.bin.signed
rm -rf /data/openpilot/pandaextra
popd

# Build stuff
export PYTHONPATH="/data/openpilot:/data/openpilot/pyextra"
SCONS_CACHE=1 scons -j3
nosetests -s selfdrive/test/test_openpilot.py

# Cleanup
find . -name '*.pyc' -delete
rm .sconsign.dblite

# Mark as prebuilt release
touch prebuilt

# Add built files to git
git add -f .
git commit --amend -m "openpilot v$VERSION"

# Push to release2-staging
git push -f public release2-staging

# Create dashcam release
git rm selfdrive/car/*/carcontroller.py

git commit -m "create dashcam release from release2"
git push -f public release2-staging:dashcam-staging
