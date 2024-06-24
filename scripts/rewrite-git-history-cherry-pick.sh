#!/bin/bash
# don't use set -e, cos we need to continue on error (in rebase)
# set -x
# set -ex
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot/
SRC_CLONE=/tmp/openpilot-clone/
SRC_ARCHIVE=/tmp/openpilot-release-archive/
OUT=/tmp/openpilot-tiny/

# INSTALL git-filter-repo
if [ ! -f /tmp/git-filter-repo ]; then
  curl -o /tmp/git-filter-repo https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
  chmod +x /tmp/git-filter-repo
fi

# MIRROR openpilot
if [ ! -d $SRC ]; then
  git clone --mirror https://github.com/commaai/openpilot.git $SRC

  cd $SRC

  echo "starting size $(du -sh .)"

  # don't see why is needed, since it's already up-to-date
  # git remote update

  # the git-filter-repo analysis is bliss - can be found in the repo root/filter-repo/analysis
  /tmp/git-filter-repo --force --analyze

  # push to archive repo
  # TODO: uncomment on final release
  # git push --mirror https://github.com/commaai/openpilot-archive.git
fi

# NOT NEEDED since we use v0.7.1 tag instead
# if [ ! -d $SRC_ARCHIVE ]; then
#   git clone --mirror https://github.com/commaai/openpilot-release-archive.git $SRC_ARCHIVE

#   cd $SRC

#   # add openpilot-release-archive as a remote
#   git remote add archive $SRC_ARCHIVE

#   # fetch the content from the archive remote
#   git fetch archive
# fi

# CHERRY-PICK all master commits over devel (v0.7.1)
if [ ! -d $SRC_CLONE ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone $SRC $SRC_CLONE

  cd $SRC_CLONE

  git checkout tags/v0.7.1
  # checkout as main, since we need master ref later
  git checkout -b main

  # rm these so we don't get conflicts later
  git rm -r cereal opendbc panda selfdrive/ui/ui
  git commit -m "removed conflicting files"

  # skip-smudge to get rid of some lfs errors that it can't find the reference of some lfs files
  # we don't care about fetching/pushing lfs right now
  git lfs install --skip-smudge --local

  # squash initial setup commits
  git cherry-pick -n -X theirs 6c33a5c..59b3d06
  git commit -m "switching to master" -m "$(git log --reverse --format=%B 6c33a5c..59b3d06)"

  # get commits we want to cherry-pick
  # will start with the next commit after #59b3d06 tools is local now
  COMMITS=$(git rev-list --reverse 59b3d06..master)

  # we need this for logging
  TOTAL_COMMITS=$(echo $COMMITS | wc -w)
  CURRENT_COMMIT_NUMBER=0

  # empty this file
  > commit-map.txt

  for COMMIT in $COMMITS; do
      CURRENT_COMMIT_NUMBER=$((CURRENT_COMMIT_NUMBER + 1))
      printf "Cherry-picking commit %d out of %d: %s\n" "$CURRENT_COMMIT_NUMBER" "$TOTAL_COMMITS" "$COMMIT"

      # extract commit metadata
      AUTHOR_NAME=$(git show -s --format='%an' $COMMIT)
      AUTHOR_EMAIL=$(git show -s --format='%ae' $COMMIT)
      COMMITTER_NAME=$(git show -s --format='%cn' $COMMIT)
      COMMITTER_EMAIL=$(git show -s --format='%ce' $COMMIT)
      AUTHOR_DATE=$(git show -s --format='%ad' $COMMIT)
      COMMIT_DATE=$(git show -s --format='%cd' $COMMIT)

      # set environment variables to preserve author/committer and dates
      export GIT_AUTHOR_NAME="$AUTHOR_NAME"
      export GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL"
      export GIT_COMMITTER_NAME="$COMMITTER_NAME"
      export GIT_COMMITTER_EMAIL="$COMMITTER_EMAIL"
      export GIT_AUTHOR_DATE="$AUTHOR_DATE"
      export GIT_COMMITTER_DATE="$COMMIT_DATE"

      # cherry-pick the commit
      if ! GIT_OUTPUT=$(git cherry-pick -m 1 -X theirs $COMMIT 2>&1); then
        # check if the failure is because of an empty commit
        if [[ "$GIT_OUTPUT" == *"The previous cherry-pick is now empty"* ]]; then
          echo "Empty commit detected. Skipping commit $COMMIT"
          git cherry-pick --skip
          # log it was empty to the mapping file
          echo "$COMMIT EMPTY" >> commit-map.txt
        else
          # handle other errors or conflicts
          echo "Cherry-pick failed. Handling error..."
          echo "$GIT_OUTPUT"
          exit 1
        fi
      else
        # capture the new commit hash
        NEW_COMMIT=$(git rev-parse HEAD)

        # save the old and new commit hashes to the mapping file
        echo "$COMMIT $NEW_COMMIT" >> commit-map.txt
      fi
  done

  # remove all old tags
  git tag -l | xargs git tag -d

  # read each line from the tag-commit-map.txt
  while IFS=' ' read -r TAG OLD_COMMIT; do
    # search for the new commit in commit-map.txt corresponding to the old commit
    NEW_COMMIT=$(grep "^$OLD_COMMIT " "commit-map.txt" | awk '{print $2}')

    # check if this is a rebased commit
    if [ -z "$NEW_COMMIT" ]; then
      # if not, then just use old commit hash
      NEW_COMMIT=$OLD_COMMIT
    fi

    printf "Recreating tag %s from commit %s\n" "$TAG" "$NEW_COMMIT"
    git tag -f "$TAG" "$NEW_COMMIT"
  done < "$DIR/tag-commit-map.txt"

  # uninstall lfs since we don't want to touch (push to) lfs right now
  # git push will also push lfs, if we don't uninstall (--local so just for this repo)
  git lfs uninstall --local

  # come back to master
  git branch -D master
  git checkout -b master
  git branch -D main

  # push to $SRC
  git push --force --set-upstream origin master

  # force push tags
  git push --tags --force
fi

# VALIDATE cherry-pick
if [ ! -f "$SRC_CLONE/commit-diff.txt" ]; then
  cd $SRC_CLONE

  TOTAL_COMMITS=$(grep -cve '^\s*$' commit-map.txt)
  CURRENT_COMMIT_NUMBER=0
  COUNT_SAME=0
  COUNT_DIFF=0
  VALIDATE_IGNORE_FILES=(
    ".github/ISSUE_TEMPLATE/bug_report.md"
    ".github/pull_request_template.md"
  )

  # empty file
  > commit-diff.txt

  echo "Validating commits"

  # will store raw diffs here, if exist
  mkdir -p differences

  # read each line from commit-map.txt
  while IFS=' ' read -r OLD_COMMIT NEW_COMMIT; do
    if [ "$NEW_COMMIT" == "EMPTY" ]; then
      continue
    fi
    CURRENT_COMMIT_NUMBER=$((CURRENT_COMMIT_NUMBER + 1))
    # retrieve short hashes and dates for the old and new commits
    OLD_COMMIT_SHORT=$(git rev-parse --short $OLD_COMMIT)
    NEW_COMMIT_SHORT=$(git rev-parse --short $NEW_COMMIT)
    OLD_DATE=$(git show -s --format='%cd' $OLD_COMMIT)
    NEW_DATE=$(git show -s --format='%cd' $NEW_COMMIT)

    echo -ne "[$CURRENT_COMMIT_NUMBER/$TOTAL_COMMITS] Comparing old commit $OLD_COMMIT_SHORT ($OLD_DATE) with new commit $NEW_COMMIT_SHORT ($NEW_DATE)"\\r
    
    # generate lists of files and their hashes for the old and new commits, excluding ignored files
    OLD_FILES=$(git ls-tree -r $OLD_COMMIT | grep -vE "$(IFS='|'; echo "${VALIDATE_IGNORE_FILES[*]}")")
    NEW_FILES=$(git ls-tree -r $NEW_COMMIT | grep -vE "$(IFS='|'; echo "${VALIDATE_IGNORE_FILES[*]}")")

    # Compare the diffs
    if diff <(echo "$OLD_FILES") <(echo "$NEW_FILES") > /dev/null; then
      # echo "Old commit $OLD_COMMIT_SHORT and new commit $NEW_COMMIT_SHORT are equivalent."
      COUNT_SAME=$((COUNT_SAME + 1))
    else
      echo "[$CURRENT_COMMIT_NUMBER/$TOTAL_COMMITS] Difference found between old commit $OLD_COMMIT_SHORT and new commit $NEW_COMMIT_SHORT" >> commit-diff.txt
      COUNT_DIFF=$((COUNT_DIFF + 1))
      set +e
      diff -u <(echo "$OLD_FILES") <(echo "$NEW_FILES") > "differences/$CURRENT_COMMIT_NUMBER-$OLD_COMMIT_SHORT-$NEW_COMMIT_SHORT"
      set -e
    fi
  done < "commit-map.txt"

  echo "Summary:" >> commit-diff.txt
  echo "Equivalent commits: $COUNT_SAME" >> commit-diff.txt
  echo "Different commits: $COUNT_DIFF" >> commit-diff.txt
fi

if [ ! -d $OUT ]; then
  cp -r $SRC $OUT

  cd $OUT

  # remove all non-master branches
  # TODO: need to see if we "redo" the other branches (except master, master-ci, devel, devel-staging, release3, release3-staging, dashcam3, dashcam3-staging, testing-closet*, hotfix-*)
  git branch | grep -v "^  master$" | grep -v "\*" | xargs git branch -D
  git for-each-ref --format='%(refname)' | grep -v 'refs/heads/master$' | xargs -I {} git update-ref -d {}

  # import "almost" everything to lfs
  git lfs migrate import --everything --include="*.dlc,*.onnx,*.svg,*.png,*.gif,*.ttf,*.wav,selfdrive/car/tests/test_models_segs.txt,system/hardware/tici/updater,selfdrive/ui/qt/spinner_larch64,selfdrive/ui/qt/text_larch64,third_party/**/*.a,third_party/**/*.so,third_party/**/*.so.*,third_party/**/*.dylib,third_party/acados/*/t_renderer,third_party/qt5/larch64/bin/lrelease,third_party/qt5/larch64/bin/lupdate,third_party/catch2/include/catch2/catch.hpp,*.apk,*.apkpatch,*.jar,*.pdf,*.jpg,*.mp3,*.thneed,*.tar.gz,*.npy,*.csv,*.a,*.so*,*.dylib,*.o,*.b64,selfdrive/hardware/tici/updater,selfdrive/boardd/tests/test_boardd,selfdrive/ui/qt/spinner_aarch64,installer/updater/updater,selfdrive/debug/profiling/simpleperf/**/*,selfdrive/hardware/eon/updater,selfdrive/ui/qt/text_aarch64,selfdrive/debug/profiling/pyflame/**/*,installer/installers/installer_openpilot,installer/installers/installer_dashcam,selfdrive/ui/text/text,selfdrive/ui/android/text/text,selfdrive/ui/spinner/spinner,selfdrive/visiond/visiond,selfdrive/loggerd/loggerd,selfdrive/sensord/sensord,selfdrive/sensord/gpsd,selfdrive/ui/android/spinner/spinner,selfdrive/ui/qt/spinner,selfdrive/ui/qt/text,_stringdefs.py,dfu-util-aarch64-linux,dfu-util-aarch64,dfu-util-x86_64-linux,dfu-util-x86_64,stb_image.h,clpeak3,clwaste,apk/**/*,external/**/*,phonelibs/**/*,third_party/boringssl/**/*,flask/**/*,panda/**/*,board/**/*,messaging/**/*,cereal/**/*,opendbc/**/*,tools/cabana/chartswidget.cc,third_party/nanovg/**/*,selfdrive/controls/lib/lateral_mpc/lib_mpc_export/**/*,selfdrive/ui/paint.cc,werkzeug/**/*,pyextra/**/*,third_party/android_hardware_libhardware/**/*,selfdrive/controls/lib/lead_mpc_lib/lib_mpc_export/**/*,selfdrive/locationd/laikad.py,selfdrive/locationd/test/test_laikad.py,tools/gpstest/test_laikad.py,selfdrive/locationd/laikad_helpers.py,tools/nui/**/*,jsonrpc/**/*,selfdrive/controls/lib/longitudinal_mpc/lib_mpc_export/**/*,selfdrive/controls/lib/lateral_mpc/mpc_export/**/*,selfdrive/camerad/cameras/camera_qcom.cc,selfdrive/manager.py,selfdrive/modeld/models/driving.cc,third_party/curl/**/*,selfdrive/modeld/thneed/debug/**/*,selfdrive/modeld/thneed/include/**/*,third_party/openmax/**/*,selfdrive/controls/lib/longitudinal_mpc/mpc_export/**/*,selfdrive/controls/lib/longitudinal_mpc_model/lib_mpc_export/**/*,Pipfile,Pipfile.lock,gunicorn/**/*,*.qm,jinja2/**/*,click/**/*,dbcs/**/*,websocket/**/*"

  # this is needed after lfs import
  git reflog expire --expire=now --all
  git gc --prune=now --aggressive

  # check the git-filter-repo analysis again - can be found in the repo root/filter-repo/analysis
  /tmp/git-filter-repo --force --analyze

  echo "new one is $(du -sh .)"
fi

cd $OUT

# fetch all lfs files from https://github.com/commaai/openpilot.git
# some lfs files are missing on gitlab, but they can be found on github
git config lfs.url https://github.com/commaai/openpilot.git/info/lfs
git config lfs.pushurl ssh://git@github.com/commaai/openpilot.git
git lfs fetch --all || true

# also fetch all lfs files from https://gitlab.com/commaai/openpilot-lfs.git
git config lfs.url https://gitlab.com/commaai/openpilot-lfs.git/info/lfs
git config lfs.pushurl ssh://git@gitlab.com/commaai/openpilot-lfs.git
git lfs fetch --all || true

# final push - will also push lfs
git push --mirror https://github.com/commaai/openpilot-tiny.git
