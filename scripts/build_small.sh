#!/usr/bin/env bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# git clone --mirror
SRC=/tmp/openpilot.git/
OUT=/tmp/smallpilot/

echo "starting size $(du -hs .git/)"

rm -rf $OUT

cd $SRC
git remote update

# copy contents
#rsync -a --exclude='.git/' $DIR $OUT

cp -r $SRC $OUT

cd $OUT

# remove all tags
git tag -l | xargs git tag -d

# remove non-master branches
BRANCHES="release2 release3 devel master-ci nightly"
for branch in $BRANCHES; do
  git branch -D $branch
  git branch -D ${branch}-staging || true
done

#git gc
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now
echo "new one is $(du -hs .)"
