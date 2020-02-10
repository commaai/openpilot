#!/data/data/com.termux/files/usr/bin/bash -e

mkdir -p /dev/shm
chmod 777 /dev/shm


add_subtree() {
  echo "[-] adding $2 subtree T=$SECONDS"
  if [ -d "$2" ]; then
    if git subtree pull --prefix "$2" https://github.com/commaai/"$1".git "$3" --squash -m "Merge $2 subtree"; then
      echo "git subtree pull succeeds"
    else
      echo "git subtree pull failed, fixing"
      git merge --abort || true
      git rm -r $2
      git commit -m "Remove old $2 subtree"
      git subtree add --prefix "$2" https://github.com/commaai/"$1".git "$3" --squash
    fi
  else
    git subtree add --prefix "$2" https://github.com/commaai/"$1".git "$3" --squash
  fi
}

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

# subtrees to make updates more reliable. updating them needs a clean tree
add_subtree "cereal" "cereal" master
add_subtree "panda" "panda" master
add_subtree "opendbc" "opendbc" master
add_subtree "openpilot-pyextra" "pyextra" master

# leave .git alone
echo "[-] erasing old openpilot T=$SECONDS"
rm -rf $TARGET_DIR/* $TARGET_DIR/.gitmodules

# dont delete our subtrees
git checkout -- cereal panda opendbc pyextra

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
PYTHONPATH="$SOURCE_DIR:$SOURCE_DIR/pyextra" nosetests -s selfdrive/test/test_openpilot.py

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
