#!/bin/bash
# don't use set -e, cos we need to continue on error (in rebase)
# set -x
# set -ex
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot/
SRC_CLONE=/tmp/openpilot-clone/
# SRC_ARCHIVE=/tmp/openpilot-release-archive/
OUT=/tmp/openpilot-tiny/

if [ ! -f /tmp/git-filter-repo ]; then
  wget -O /tmp/git-filter-repo https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
  chmod +x /tmp/git-filter-repo
fi

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

# if [ ! -d $SRC_ARCHIVE ]; then
#   git clone --mirror https://github.com/commaai/openpilot-release-archive.git $SRC_ARCHIVE

#   cd $SRC

#   # add openpilot-release-archive as a remote
#   git remote add archive $SRC_ARCHIVE

#   # fetch the content from the archive remote
#   git fetch archive
# fi

if [ ! -d $SRC_CLONE ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone $SRC $SRC_CLONE

  cd $SRC_CLONE

  git checkout tags/v0.7.1
  # checkout as main, since we need master ref later
  git checkout -b main

  # rm these so that we don't get an error when adding them as submodules later
  git rm -r cereal opendbc panda
  git commit -m "removed cereal opendbc panda"

  # skip-smudge to get rid of some lfs errors that it can't find the reference of some lfs files
  # we don't care about fetching/pushing lfs right now
  git lfs install --skip-smudge --local

  # get commits we want to cherry-pick
  COMMITS=$(git rev-list --reverse 6c33a5c..master)

  # we need this for logging
  TOTAL_COMMITS=$(echo $COMMITS | wc -w)
  CURRENT_COMMIT_NUMBER=0

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
      git cherry-pick -m 1 --keep-redundant-commits -X theirs $COMMIT
  done

  # uninstall lfs since we don't want to touch (push to) lfs right now
  # git push will also push lfs, if we don't uninstall (--local so just for this repo)
  git lfs uninstall --local

  # come back to master
  git branch -D master
  git checkout -b master
  git branch -D main

  # push to $SRC
  git push --force
fi

if [ ! -d $OUT ]; then
  cp -r $SRC $OUT

  cd $OUT

  # remove all tags
  # TODO: we need to keep the tags, WIP on how we actually do it
  # git tag -l | xargs git tag -d

  # remove all non-master branches
  # TODO: need to see if we "redo" the other branches (except master, master-ci, devel, devel-staging, release3, release3-staging, dashcam3, dashcam3-staging, testing-closet*, hotfix-*)
  git branch | grep -v "^  master$" | grep -v "\*" | xargs git branch -D
  git for-each-ref --format='%(refname)' | grep -v 'refs/heads/master$' | xargs -I {} git update-ref -d {}

  # import almost everything to lfs
  # WIP still needs to add some/more files
  git lfs migrate import --everything --include="*.dlc,*.onnx,*.svg,*.png,*.gif,*.ttf,*.wav,selfdrive/car/tests/test_models_segs.txt,system/hardware/tici/updater,selfdrive/ui/qt/spinner_larch64,selfdrive/ui/qt/text_larch64,third_party/**/*.a,third_party/**/*.so,third_party/**/*.so.*,third_party/**/*.dylib,third_party/acados/*/t_renderer,third_party/qt5/larch64/bin/lrelease,third_party/qt5/larch64/bin/lupdate,third_party/catch2/include/catch2/catch.hpp,*.apk,*.apkpatch,*.jar,*.pdf,*.jpg,*.mp3,*.ico,*.thneed,*.tar.gz,*.npy,*.csv,*.a,*.so*,*.dylib,*.o,*.b64,selfdrive/hardware/tici/updater,selfdrive/boardd/tests/test_boardd,selfdrive/ui/qt/spinner_aarch64,installer/updater/updater,selfdrive/debug/profiling/simpleperf/bin/android/arm64/simpleperf,selfdrive/hardware/eon/updater,selfdrive/ui/qt/text_aarch64,selfdrive/debug/profiling/pyflame/pyflame,installer/installers/installer_openpilot,installer/installers/installer_dashcam,selfdrive/ui/text/text,selfdrive/ui/android/text/text,selfdrive/ui/spinner/spinner,selfdrive/visiond/visiond,selfdrive/loggerd/loggerd,selfdrive/sensord/sensord,selfdrive/sensord/gpsd,selfdrive/ui/android/spinner/spinner,selfdrive/ui/qt/spinner,selfdrive/ui/qt/text,_stringdefs.py,dfu-util-aarch64-linux,dfu-util-aarch64,dfu-util-x86_64-linux,dfu-util-x86_64,stb_image.h,clpeak3,clwaste,apk/**/*,external/**/*,phonelibs/**/*,third_party/boringssl/**/*,flask/**/*,panda/**/*,board/**/*,messaging/**/*,cereal/**/*,opendbc/**/*,tools/cabana/chartswidget.cc,third_party/nanovg/**/*,selfdrive/controls/lib/lateral_mpc/lib_mpc_export/**/*,selfdrive/ui/paint.cc,werkzeug/**/*,pyextra/**/*,third_party/android_hardware_libhardware/**/*,selfdrive/controls/lib/lead_mpc_lib/lib_mpc_export/**/*,selfdrive/locationd/laikad.py,selfdrive/locationd/test/test_laikad.py,tools/gpstest/test_laikad.py,selfdrive/locationd/laikad_helpers.py,tools/nui/**/*,jsonrpc/**/*,selfdrive/controls/lib/longitudinal_mpc/lib_mpc_export/**/*,selfdrive/controls/lib/lateral_mpc/mpc_export/**/*,selfdrive/camerad/cameras/camera_qcom.cc,selfdrive/manager.py,selfdrive/modeld/models/driving.cc,third_party/curl/**/*,selfdrive/modeld/thneed/debug/**/*,selfdrive/modeld/thneed/include/**/*,third_party/openmax/**/*"

  # this is needed after lfs import
  git reflog expire --expire=now --all
  git gc --prune=now --aggressive

  # don't delete external and phonelibs anymore
  # git-filter-repo doesn't seem to need reflog and gc after the command
  # /tmp/git-filter-repo --invert-paths --path external/ --path phonelibs/

  # check the git-filter-repo analysis again - can be found in the repo root/filter-repo/analysis
  /tmp/git-filter-repo --force --analyze

  echo "new one is $(du -sh .)"
fi

cd $OUT

# fetch all lfs files from https://github.com/commaai/openpilot.git
# some lfs files are missing on gitlab, but they can be found on github
git config lfs.url https://github.com/commaai/openpilot.git/info/lfs
git config lfs.pushurl ssh://git@github.com/commaai/openpilot.git
git lfs fetch --all

# also fetch from gitlab
git config lfs.url https://gitlab.com/commaai/openpilot-lfs.git/info/lfs
git config lfs.pushurl ssh://git@gitlab.com/commaai/openpilot-lfs.git
git lfs fetch --all

# new lfs urls for testing repo (these should be removed)
git config lfs.url https://gitlab.com/andiradulescu/openpilot-tiny-cherry-pick.git/info/lfs
git config lfs.pushurl ssh://git@gitlab.com/andiradulescu/openpilot-tiny-cherry-pick.git

# final push - will also push lfs
# git push --mirror git@gitlab.com:andiradulescu/openpilot-tiny-cherry-pick.git
git push --mirror git@github.com:andiradulescu/openpilot-tiny-cherry-pick.git
# git push --mirror https://github.com/commaai/openpilot-tiny.git
