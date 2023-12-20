#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot.git/
OUT=/tmp/smallpilot/

if [ ! -d $SRC ]; then
  git clone --mirror https://github.com/commaai/openpilot.git $SRC
fi

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

# remove all non-master branches
git branch | grep -v "^  master$" | xargs git branch -D

# remove all the junk
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now
echo "before is $(du -hs .)"

# run the bfg
bfg="java -jar ~/Downloads/bfg.jar"
$bfg --strip-blobs-bigger-than 10M $OUT

git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now
echo "new one is $(du -hs .)"
