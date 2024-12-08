#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

git clone --bare https://github.com/commaai/openpilot
cp -r openpilot.git openpilot_backup
cd openpilot.git

# backup old repo
git push git@github.com:commaai/openpilot-archive.git +refs/heads/master:refs/heads/master
git push git@github.com:commaai/openpilot-archive.git +refs/heads/*:refs/heads/*
git push git@github.com:commaai/openpilot-archive.git +refs/tags/*:refs/tags/*
git push --mirror git@github.com:commaai/openpilot-archive.git

# ignore all release branches
git for-each-ref --format='delete %(refname)' | grep 'dashcam3\|devel\|master-ci\|nightly\|release2\|release3\|release3-staging' | git update-ref --stdin

# re-tag old releases on master
declare -A TAGS=( ["f8cb04e4a8b032b72a909f68b808a50936184bee"]="v0.9.7" ["0b4d08fab8e35a264bc7383e878538f8083c33e5"]="v0.9.6" ["3b1e9017c560499786d8a0e46aaaeea65037acac"]="v0.9.5" ["fa310d9e2542cf497d92f007baec8fd751ffa99c"]="v0.9.4" ["8704c1ff952b5c85a44f50143bbd1a4f7b4887e2"]="v0.9.3" ["c7d3b28b93faa6c955fb24bc64031512ee985ee9"]="v0.9.2" ["89f68bf0cbf53a81b0553d3816fdbe522f941fa1"]="v0.9.1" ["58b84fb401a804967aa0dd5ee66fafa90194fd30"]="v0.9.0" ["f41dc62a12cc0f3cb8c5453c0caa0ba21e1bd01e"]="v0.8.16" ["5a7c2f90361e72e9c35e88abd2e11acdc4aba354"]="v0.8.15" ["71901c94dbbaa2f9f156a80c14cc7ea65219fc7c"]="v0.8.14" ["95da47079510afc91665263619e5939126da637c"]="v0.8.13" ["472177e2a8a1d002e56f9096326fd2dff62e54f9"]="v0.8.12" ["08078acbd0b4f7da469c7dff6159000e358974a9"]="v0.8.11" ["687925c775c375495f9827946138a724bde00b9d"]="v0.8.10" ["204e5a090735a059d69c29145a4cee49450da07e"]="v0.8.9" ["4be956f8861ecbb521ef9503a3c87b07c9d36721"]="v0.8.8" ["589f82c76627d634761a31a34b2488403556eb0b"]="v0.8.7" ["507cfc8910f74ddb8810039d68b880b426ff9ff9"]="v0.8.6" ["d47b00b45a866bef088f51d1ff31de5885ab04e9"]="v0.8.5" ["553e7d1cce314e7eb0587186b1764c3ff43bed62"]="v0.8.4" ["9896438d1511602a1ff87f7c4eb3c7172b30104a"]="v0.8.3" ["280192ed1443f112463417c2d815ea8ee2762fbd"]="v0.8.2" ["8039361567e4659eae2a084e6f39f34acadf4cac"]="v0.8.1" ["d56e04c0d960c8d3d4ab88b578dc508a2b4e07dc"]="v0.8" ["3d456e5d0fbf0c9887d0499dee812f2b029edf6d"]="v0.7.10" ["81763a18b5d0e379b749e090ecce36a91fca7c43"]="v0.7.9" ["9bc0b350fd273bbb2deb3dcaef0312944e4f6cfd"]="v0.7.8" ["ede5b632b58c55e4ff003f948efae07fe03c2280"]="v0.7.7" ["775acd11ba2e0a8c2f5a5655338718d796491b36"]="v0.7.6.1" ["302417b4cf0dcf00d45e4995b5410e543ad121d1"]="v0.7.5" ["12ff088b42221dd17d9d97decb1fc61a7cb0a861"]="v0.7.4" ["9563f7730252451fdcba9bc3d9fe36dab9c86a26"]="v0.7.3" ["8321cf283abbc2ca3fda7e0c7a069a77a492fe0c"]="v0.7.2" ["1e1de64a1e59476b7b3d3558b92149246d5c3292"]="v0.7.1" ["a2ae18d1dbd1e59c38ce22fa25ddffbd1d3084e3"]="v0.7" ["d4eb5a6eafdd4803d09e6f3963918216cca5a81f"]="v0.6.6" ["70d17cd69b80e7627dcad8fd5b6438f2309ac307"]="v0.6.5" ["58f376002e0c654fbc2de127765fa297cf694a33"]="v0.6.4" ["d5f9caa82d80cdcc7f1b7748f2cf3ccbf94f82a3"]="v0.6.3" ["095ef5f9f60fca1b269aabcc3cfd322b17b9e674"]="v0.6.2" ["cf5c4aeacb1703d0ffd35bdb5297d3494fee9a22"]="v0.6.1" ["60a20537c5f3fcc7f11946d81aebc8f90c08c117"]="v0.6" ["dd34ccfe288ebda8e2568cf550994ae890379f45"]="v0.5.13" ["3f9059fea886f1fa3b0c19a62a981d891dcc84eb"]="v0.5.12" ["2f92d577f995ff6ae1945ef6b89df3cb69b92999"]="v0.5.11" ["5a9d89ed42ddcd209d001a10d7eb828ef0e6d9de"]="v0.5.10" ["0207a970400ee28d3e366f2e8f5c551281accf02"]="v0.5.9" ["b967da5fc1f7a07e3561db072dd714d325e857b0"]="v0.5.8" ["210db686bb89f8696aa040e6e16de65424b808c9"]="v0.5.7" ["860a48765d1016ba226fb2c64aea35a45fe40e4a"]="v0.5.6" ["8f3539a27b28851153454eb737da9624cccaed2d"]="v0.5.5" ["a422246dc30bce11e970514f13f7c110f4470cc3"]="v0.5.4" ["285c52eb693265a0a530543e9ca0aeb593a2a55e"]="v0.5.3" ["0129a8a4ff8da5314e8e4d4d3336e89667ff6d54"]="v0.5.2" ["6f3d10a4c475c4c4509f0b370805419acd13912d"]="v0.5.1" ["de33bc46452b1046387ee2b3a03191b2c71135fb"]="v0.5" ["ae5cb7a0dab8b1bed9d52292f9b4e8e66a0f8ec9"]="v0.4.7" ["c6df34f55ba8c5a911b60d3f9eb20e3fa45f68c1"]="v0.4.6" ["37285038d3f91fa1b49159c4a35a8383168e644f"]="v0.4.5" ["9a9ff839a9b70cb2601d7696af743f5652395389"]="v0.4.4" ["28c0797d30175043bbfa31307b63aab4197cf996"]="v0.4.2" ["4474b9b3718653aeb0aee26422caefb90460cc0e"]="v0.4.1" ["da52d065a4c4f52d6017a537f3a80326f5af8bdc"]="v0.4.0.2" ["9d3963559ae7b15193057937ff3e72481899f40d"]="v0.3.5"  ["1b8c44b5067525a5d266b6e99799d8097da76a29"]="v0.3.4" ["5cf91d0496688fed4f2a6c7021349b1fc0e057a2"]="v0.3.3" ["7fe46f1e1df5dec08a940451ba0feefd5c039165"]="v0.3.2" ["41e3a0f699f5c39cb61a15c0eb7a4aa816d47c24"]="v0.3.1" ["c5d8aec28b5230d34ae4b677c2091cc3dec7e3e8"]="v0.3.0" ["693bcb0f83478f2651db6bac9be5ca5ad60d03f3"]="v0.2.9" ["95a349abcc050712c50d4d85a1c8a804eee7f6c2"]="v0.2.8" ["c6ba5dc5391d3ca6cda479bf1923b88ce45509a0"]="v0.2.7" ["6c3afeec0fb439070b2912978b8dbb659033b1d9"]="v0.2.6" ["29c58b45882ac79595356caf98580c1d2a626011"]="v0.2.5" ["ecc565aa3fdc4c7e719aadc000e1fdc4d80d4fe0"]="v0.2.4" ["adaa4ed350acda4067fc0b455ad15b54cdf4c768"]="v0.2.3" ["a64b9aa9b8cb5863c917b6926516291a63c02fe5"]="v0.2.2" ["17d9becd3c673091b22f09aa02559a9ed9230f50"]="v0.2.1" ["449b482cc3236ccf31829830b4f6a44b2dcc06c2"]="v0.2" ["e94a30bec07e719c5a7b037ca1f4db8312702cce"]="v0.1" )
for tag in "${!TAGS[@]}"; do git tag -f "${TAGS[$tag]}" "$tag" ; done

# get master root commit
ROOT_COMMIT=$(git rev-list --max-parents=0 HEAD | tail -n 1)

# link master and devel
git replace --graft $ROOT_COMMIT v0.7.1
git-filter-repo --prune-empty never --force --commit-callback 'h=commit.original_id.decode("utf-8");m=commit.message.decode("utf-8");commit.message=str.encode(m + "\n" + "old-commit-hash: " + h)'

# delete replace refs
git for-each-ref --format='delete %(refname)' refs/replace | git update-ref --stdin

# machine validation
tail -n +2 "filter-repo/commit-map" | tr ' ' '\n' | xargs -P $(nproc) -n 2 bash -c 'H1=$(cd ../openpilot_backup && git ls-tree -r $0 | sha1sum) && H2=$(git ls-tree -r $1 | sha1sum) && echo "$H1 $H2" >> /tmp/GIT_HASHES && diff <(echo $H1) <(echo $H2) || exit 255'
# human validation
less /tmp/GIT_HASH

# cleanup
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# get all lfs files
set +e
git config lfs.url https://github.com/commaai/openpilot.git/info/lfs
git lfs fetch --all
git config lfs.url https://gitlab.com/commaai/openpilot-lfs.git/info/lfs
git lfs fetch --all
set -e

# add new files to lfs
git lfs migrate import --everything --include="*.ico,*.dlc,*.onnx,*.svg,*.png,*.gif,*.ttf,*.wav,system/hardware/tici/updater,selfdrive/ui/qt/spinner_larch64,selfdrive/ui/qt/text_larch64,third_party/**/*.a,third_party/**/*.so,third_party/**/*.so.*,third_party/**/*.dylib,third_party/acados/*/t_renderer,third_party/qt5/larch64/bin/lrelease,third_party/qt5/larch64/bin/lupdate,third_party/catch2/include/catch2/catch.hpp,*.apk,*.apkpatch,*.jar,*.pdf,*.jpg,*.mp3,*.thneed,*.tar.gz,*.npy,*.csv,*.a,*.so*,*.dylib,*.o,*.b64,selfdrive/hardware/tici/updater,selfdrive/boardd/tests/test_boardd,selfdrive/ui/qt/spinner_aarch64,installer/updater/updater,selfdrive/debug/profiling/simpleperf/**/*,selfdrive/hardware/eon/updater,selfdrive/ui/qt/text_aarch64,selfdrive/debug/profiling/pyflame/**/*,installer/installers/installer_openpilot,installer/installers/installer_dashcam,selfdrive/ui/text/text,selfdrive/ui/android/text/text,selfdrive/ui/spinner/spinner,selfdrive/visiond/visiond,selfdrive/loggerd/loggerd,selfdrive/sensord/sensord,selfdrive/sensord/gpsd,selfdrive/ui/android/spinner/spinner,selfdrive/ui/qt/spinner,selfdrive/ui/qt/text,_stringdefs.py,dfu-util-aarch64-linux,dfu-util-aarch64,dfu-util-x86_64-linux,dfu-util-x86_64,stb_image.h,clpeak3,clwaste,apk/**/*,external/**/*,phonelibs/**/*,third_party/boringssl/**/*,pyextra/**/*,panda/board/**/inc/*.h,panda/board/obj/*.elf,board/inc/*.h,third_party/nanovg/**/*,selfdrive/controls/lib/lateral_mpc/lib_mpc_export/**/*,pyextra/**/*,third_party/android_hardware_libhardware/**/*,selfdrive/controls/lib/lead_mpc_lib/lib_mpc_export/**/*,*.pro,selfdrive/controls/lib/longitudinal_mpc/lib_mpc_export/**/*,selfdrive/controls/lib/lateral_mpc/mpc_export/**/*,third_party/curl/**/*,selfdrive/modeld/thneed/debug/**/*,selfdrive/modeld/thneed/include/**/*,third_party/openmax/**/*,selfdrive/controls/lib/longitudinal_mpc/mpc_export/**/*,selfdrive/controls/lib/longitudinal_mpc_model/lib_mpc_export/**/*,Pipfile,Pipfile.lock,poetry.lock,*.qm"

# set new lfs endpoint
git config lfs.url https://gitlab.com/commaai/openpilot-lfs.git/info/lfs
git config lfs.pushurl ssh://git@gitlab.com/commaai/openpilot-lfs.git

# push all branch+tag (scary stuff...)
git push -f --set-upstream git@github.com:commaai/openpilot.git +refs/heads/*:refs/heads/* +refs/tags/*:refs/tags/*
