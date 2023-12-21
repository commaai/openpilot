#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot.git/
OUT=/tmp/smallpilot/

if [ ! -d $SRC ]; then
  git clone --bare --mirror https://github.com/commaai/openpilot.git $SRC
fi

echo "starting size $(du -hs .git/)"

rm -rf $OUT

cd $SRC
git remote update

# push to archive repo
#git push -f --mirror https://github.com/commaai/openpilot-archive.git

# copy contents
#rsync -a --exclude='.git/' $DIR $OUT

cp -r $SRC $OUT

cd $OUT

# remove all tags
git tag -l | xargs git tag -d

# remove all non-master branches
git branch | grep -v "^  master$" | grep -v "\*" | xargs git branch -D
git for-each-ref --format='%(refname)' | grep -v 'refs/heads/master$' | xargs -I {} git update-ref -d {}

# remove all the junk
#git reflog expire --expire=now --all
#git gc --prune=now
#git gc --aggressive --prune=now
#echo "before is $(du -hs .)"

# delete junk files
bfg="java -jar $HOME/Downloads/bfg.jar"
$bfg --strip-blobs-bigger-than 100K $OUT
#git filter-repo --force --blob-callback $DIR/delete.py

#wget -O /tmp/git-filter-repo https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
#chmod +x /tmp/git-filter-repo
#/tmp/git-filter-repo -h

git reflog expire --expire=now --all
git gc --prune=now --aggressive
echo "new one is $(du -hs .)"

cd $OUT
git push -f --mirror https://github.com/commaai/openpilot-tiny.git
