#!/usr/bin/env bash
set -e

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"

export GIT_SSH_COMMAND="ssh -i /data/gitkey"

# set CLEAN to build outside of CI
if [ ! -z "$CLEAN" ]; then
  # Create folders
  rm -rf /data/openpilot
  mkdir -p /data/openpilot
  cd /data/openpilot

  # Create git repo
  git init
  git remote add origin git@github.com:commaai/openpilot.git
  git fetch origin devel-staging
else
  cd /data/openpilot
  git clean -xdf
  git branch -D release2-staging || true
fi

git fetch origin release2-staging
git fetch origin dashcam-staging

# Create release2 with no history
if [ ! -z "$CLEAN" ]; then
  git checkout --orphan release2-staging origin/devel-staging
else
  git checkout --orphan release2-staging
fi

VERSION=$(cat selfdrive/common/version.h | awk -F[\"-]  '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-release\"" > selfdrive/common/version.h

git commit -m "openpilot v$VERSION"

# Build signed panda firmware
pushd panda/
CERT=/tmp/pandaextra/certs/release RELEASE=1 scons
mv board/obj/panda.bin.signed /tmp/panda.bin.signed
popd

# Build stuff
ln -sfn /data/openpilot /data/pythonpath
export PYTHONPATH="/data/openpilot:/data/openpilot/pyextra"
SCONS_CACHE=1 scons -j3

# Run tests
python selfdrive/manager/test/test_manager.py
selfdrive/car/tests/test_car_interfaces.py

# Cleanup
find . -name '*.a' -delete
find . -name '*.o' -delete
find . -name '*.os' -delete
find . -name '*.pyc' -delete
find . -name '__pycache__' -delete
rm -rf panda/board panda/certs panda/crypto
rm -rf .sconsign.dblite Jenkinsfile release/

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

# Print committed files that are normally gitignored
#git status --ignored

if [ ! -z "$PUSH" ]; then
  git remote set-url origin git@github.com:commaai/openpilot.git

  # Push to release2-staging
  git push -f origin release2-staging

  # Create dashcam release
  git rm selfdrive/car/*/carcontroller.py

  git commit -m "create dashcam release from release2"
  git push -f origin release2-staging:dashcam-staging
fi
