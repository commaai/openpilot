#!/usr/bin/env bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

SOURCE_DIR="$(git -C $DIR rev-parse --show-toplevel)"
if [ -z "$TARGET_DIR" ]; then
  TARGET_DIR="$(mktemp -d)"
fi

# set git identity
source $DIR/identity.sh

echo "[-] Setting up target repo T=$SECONDS"

rm -rf $TARGET_DIR
mkdir -p $TARGET_DIR
cd $TARGET_DIR
cp -r $SOURCE_DIR/.git $TARGET_DIR

echo "[-] setting up stripped branch sync T=$SECONDS"
cd $TARGET_DIR

# tmp branch
git checkout --orphan tmp

# remove everything except .git
echo "[-] erasing old openpilot T=$SECONDS"
git submodule deinit -f --all
git rm -rf --cached .
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;

# cleanup before the copy
cd $SOURCE_DIR
git clean -xdff
git submodule foreach --recursive git clean -xdff

# do the files copy
echo "[-] copying files T=$SECONDS"
cd $SOURCE_DIR
cp -pR --parents $(./release/release_files.py) $TARGET_DIR/

# in the directory
cd $TARGET_DIR
rm -rf .git/modules/
rm -f panda/board/obj/panda.bin.signed

# include source commit hash and build date in commit
GIT_HASH=$(git --git-dir=$SOURCE_DIR/.git rev-parse HEAD)
GIT_COMMIT_DATE=$(git --git-dir=$SOURCE_DIR/.git show --no-patch --format='%ct %ci' HEAD)
DATETIME=$(date '+%Y-%m-%dT%H:%M:%S')
VERSION=$(cat $SOURCE_DIR/common/version.h | awk -F\" '{print $2}')

echo -n "$GIT_HASH" > git_src_commit
echo -n "$GIT_COMMIT_DATE" > git_src_commit_date

echo "[-] committing version $VERSION T=$SECONDS"
git add -f .
git status
git commit -a -m "openpilot v$VERSION release

date: $DATETIME
master commit: $GIT_HASH
"

# should be no submodules or LFS files
git submodule status
if [ ! -z "$(git lfs ls-files)" ]; then
  echo "LFS files detected!"
  exit 1
fi

# ensure files are within GitHub's limit
BIG_FILES="$(find . -type f -not -path './.git/*' -size +95M)"
if [ ! -z "$BIG_FILES" ]; then
  printf '\n\n\n'
  echo "Found files exceeding GitHub's 100MB limit:"
  echo "$BIG_FILES"
  exit 1
fi

if [ ! -z "$BRANCH" ]; then
  echo "[-] Pushing to $BRANCH T=$SECONDS"
  git push -f origin tmp:$BRANCH
fi

echo "[-] done T=$SECONDS, ready at $TARGET_DIR"
