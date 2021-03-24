#!/usr/bin/bash -e

SOURCE_DIR=/data/openpilot_source
TARGET_DIR=/data/openpilot

ln -sf $TARGET_DIR /data/pythonpath

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"
export GIT_SSH_COMMAND="ssh -i /data/gitkey"

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

git checkout -f --track origin/master-ci
git reset --hard master-ci
git checkout master-ci
git reset --hard origin/devel
git clean -xdf

# remove everything except .git
echo "[-] erasing old openpilot T=$SECONDS"
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;

# reset tree and get version
cd $SOURCE_DIR
git clean -xdf
git checkout -- selfdrive/common/version.h

VERSION=$(cat selfdrive/common/version.h | awk -F\" '{print $2}')
echo "#define COMMA_VERSION \"$VERSION-$(git --git-dir=$SOURCE_DIR/.git rev-parse --short HEAD)-$(date '+%Y-%m-%dT%H:%M:%S')\"" > selfdrive/common/version.h

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

if [ ! -z "$CI_PUSH" ]; then
  echo "[-] Pushing to $CI_PUSH T=$SECONDS"
  git remote set-url origin git@github.com:commaai/openpilot.git
  git push -f origin master-ci:$CI_PUSH
fi

echo "[-] done T=$SECONDS"
