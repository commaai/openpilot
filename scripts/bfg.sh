#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

SRC=/tmp/openpilot.git/
OUT=/tmp/smallpilot/

wget -O /tmp/git-filter-repo https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
chmod +x /tmp/git-filter-repo

if [ ! -d $SRC ]; then
  git clone --bare --mirror https://github.com/commaai/openpilot.git $SRC
fi

cd $SRC

echo "starting size $(du -hs .)"

git remote update

/tmp/git-filter-repo --force --analyze

# push to archive repo
#git push -f --mirror https://github.com/commaai/openpilot-archive.git

# copy contents
#rsync -a --exclude='.git/' $DIR $OUT

rm -rf $OUT
cp -r $SRC $OUT

cd $OUT

# remove all tags
# git tag -l | xargs git tag -d

# remove all non-master branches
git branch | grep -v "^  master$" | grep -v "\*" | xargs git branch -D
git for-each-ref --format='%(refname)' | grep -v 'refs/heads/master$' | xargs -I {} git update-ref -d {}

# remove all the junk
#git reflog expire --expire=now --all
#git gc --prune=now
#git gc --aggressive --prune=now
#echo "before is $(du -hs .)"

# delete junk files
# bfg="java -jar $HOME/Downloads/bfg.jar"
# $bfg --strip-blobs-bigger-than 100K $OUT
# git filter-repo --force --blob-callback $DIR/delete.py

git lfs migrate import --everything --include="*.dlc,*.onnx,*.svg,*.png,*.gif,*.ttf,*.wav,selfdrive/car/tests/test_models_segs.txt,system/hardware/tici/updater,selfdrive/ui/qt/spinner_larch64,selfdrive/ui/qt/text_larch64,third_party/**/*.a,third_party/**/*.so,third_party/**/*.so.*,third_party/**/*.dylib,third_party/acados/*/t_renderer,third_party/qt5/larch64/bin/lrelease,third_party/qt5/larch64/bin/lupdate,third_party/catch2/include/catch2/catch.hpp,*.apk,*.thneed,selfdrive/hardware/tici/updater,selfdrive/boardd/tests/test_boardd,common/geocode/rg_cities1000.csv,selfdrive/ui/qt/spinner_aarch64,installer/updater/updater,selfdrive/locationd/test/ubloxRaw.tar.gz,selfdrive/debug/profiling/simpleperf/bin/android/arm64/simpleperf,selfdrive/hardware/eon/updater,selfdrive/ui/qt/text_aarch64,selfdrive/debug/profiling/pyflame/pyflame,installer/installers/installer_openpilot,installer/installers/installer_dashcam,selfdrive/ui/text/text,selfdrive/ui/android/text/text,selfdrive/ui/qt/spinnerselfdrive/locationd/kalman/helpers/chi2_lookup_table.npy,selfdrive/locationd/kalman/chi2_lookup_table.npy,selfdrive/ui/spinner/spinner"

git reflog expire --expire=now --all
git gc --prune=now --aggressive

/tmp/git-filter-repo --invert-paths --path external/ --path phonelibs/

/tmp/git-filter-repo --force --analyze

echo "new one is $(du -hs .)"

# cd $OUT
# git push -f --mirror https://github.com/commaai/openpilot-tiny.git
# git push -f --mirror git@github.com:andiradulescu/openpilot-tiny.git
# git push -f --mirror git@gitlab.com:andiradulescu/openpilot-tiny.git
