#!/bin/bash
# don't use set -e, cos we need to continue on error (in rebase)
# set -x
# set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot/
SRC_CLONE=/tmp/openpilot-clone/
# SRC_ARCHIVE=/tmp/openpilot-release-archive/
OUT=/tmp/smallpilot/

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
  git clone $SRC $SRC_CLONE

  cd $SRC_CLONE

  # seems we might not need the archive repo at all - since we already have the history of the tags in the main repo
  # git checkout --track origin/devel

  git checkout tags/v0.7.1
  git checkout -b archive

  # rm these so that we don't get an error when adding them as submodules later
  # should work now with new handle conflicts
  # git rm -r cereal opendbc panda
  # git commit -m "removed unmergeable files and directories before merge"

  # skip-smudge to get rid of some lfs errors that it can't find the reference of some lfs files
  # we don't care about fetching/pushing lfs right now
  git lfs install --skip-smudge --local

  # go to master
  git checkout master

  # rebase previous "devel" history
  # WIP - this doesn't complete, it currently errors after ~20 commits rebased
  # git rebase -X ours archive

  # new solution to rebase
  handle_conflicts() {
    # If there's a conflict, find deleted files and delete them
    git status --porcelain | grep '^UD' | cut -c4- | while read -r file; do
      git rm -- "$file"
    done
    # Add all changes (this will mark conflicts as resolved)
    git add -u
  }

  git rebase -X ours --committer-date-is-author-date archive

  # loop git rebase --continue until we are done
  while true; do
    GIT_EDITOR=true git rebase --continue

    # check the exit status of the rebase command
    if [ $? -eq 0 ]; then
      # if the rebase was successful, break out of the loop
      break
    else
      # if there was an error (e.g., a conflict), handle the conflict
      handle_conflicts
    fi
  done

  # uninstall lfs since we don't want to touch (push to) lfs right now
  # git push will also push lfs, if we don't uninstall (--local so just for this repo)
  git lfs uninstall --local

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
  git lfs migrate import --everything --include="*.dlc,*.onnx,*.svg,*.png,*.gif,*.ttf,*.wav,selfdrive/car/tests/test_models_segs.txt,system/hardware/tici/updater,selfdrive/ui/qt/spinner_larch64,selfdrive/ui/qt/text_larch64,third_party/**/*.a,third_party/**/*.so,third_party/**/*.so.*,third_party/**/*.dylib,third_party/acados/*/t_renderer,third_party/qt5/larch64/bin/lrelease,third_party/qt5/larch64/bin/lupdate,third_party/catch2/include/catch2/catch.hpp,*.apk,*.thneed,selfdrive/hardware/tici/updater,selfdrive/boardd/tests/test_boardd,common/geocode/rg_cities1000.csv,selfdrive/ui/qt/spinner_aarch64,installer/updater/updater,selfdrive/locationd/test/ubloxRaw.tar.gz,selfdrive/debug/profiling/simpleperf/bin/android/arm64/simpleperf,selfdrive/hardware/eon/updater,selfdrive/ui/qt/text_aarch64,selfdrive/debug/profiling/pyflame/pyflame,installer/installers/installer_openpilot,installer/installers/installer_dashcam,selfdrive/ui/text/text,selfdrive/ui/android/text/text,selfdrive/ui/qt/spinnerselfdrive/locationd/kalman/helpers/chi2_lookup_table.npy,selfdrive/locationd/kalman/chi2_lookup_table.npy,selfdrive/ui/spinner/spinner"

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
git config lfs.url https://gitlab.com/andiradulescu/openpilot-tiny.git/info/lfs
git config lfs.pushurl ssh://git@gitlab.com/andiradulescu/openpilot-tiny.git

# final push - will also push lfs
# git push --mirror git@gitlab.com:andiradulescu/openpilot-tiny.git
git push --mirror git@github.com:andiradulescu/openpilot-tiny.git
# git push --mirror https://github.com/commaai/openpilot-tiny.git
