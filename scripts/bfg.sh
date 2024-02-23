#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot/
SRC_ARCHIVE=/tmp/openpilot-release-archive/
OUT=/tmp/smallpilot/

wget -O /tmp/git-filter-repo https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
chmod +x /tmp/git-filter-repo

if [ ! -d $SRC ]; then
  git clone --bare --mirror https://github.com/commaai/openpilot.git $SRC
fi

if [ ! -d $SRC_ARCHIVE ]; then
  git clone --bare --mirror https://github.com/commaai/openpilot-release-archive.git $SRC_ARCHIVE
fi

cd $SRC

echo "starting size $(du -sh .)"

git remote update

# the git-filter-repo analysis is bliss - can be found in the repo root/filter-repo/analysis
/tmp/git-filter-repo --force --analyze

# Add openpilot-release-archive as a remote
git remote add archive $SRC_ARCHIVE

# Fetch the content from the archive remote
git fetch archive

# git checkout --track origin/devel
# WIP: seems we might not need the archive repo at all - since we already have the history of the tags in the main repo
git checkout tags/v0.7.1
git checkout -b archive

# rm these so that we don't get an error when adding them as submodules later
git rm -r cereal opendbc panda
git commit -m "removed unmergeable files and directories before merge"

# skip-smudge to get rid of some lfs errors that it can't find the reference of some lfs files
git lfs install --skip-smudge --local

# go to master
git checkout master

# rebase previous "devel" history
# WIP - this doesn't complete, it currently errors after ~20 commits rebased
git rebase -X ours archive

# push to archive repo
# WIP - push it when rebase is done without errors
#git push -f --mirror https://github.com/commaai/openpilot-archive.git

rm -rf $OUT
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

# don't delete external and phonelibs anymore (?!)
# git-filter-repo doesn't seem to need reflog and gc after the command
# /tmp/git-filter-repo --invert-paths --path external/ --path phonelibs/

# check the git-filter-repo analysis again - can be found in the repo root/filter-repo/analysis
/tmp/git-filter-repo --force --analyze

echo "new one is $(du -sh .)"

# uncomment this when we reach this point
# cd $OUT
# git push -f --mirror https://github.com/commaai/openpilot-tiny.git
# git push -f --mirror git@github.com:andiradulescu/openpilot-tiny.git
# git push -f --mirror git@gitlab.com:andiradulescu/openpilot-tiny.git
