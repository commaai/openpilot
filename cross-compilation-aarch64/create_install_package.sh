#!/bin/bash -e

git --version 2>&1 >/dev/null
GIT_IS_AVAILABLE=$?

if [ $GIT_IS_AVAILABLE -eq 0 ]; then 
    openpilot_dir=$(git rev-parse --show-toplevel)
else
    openpilot_dir="$HOME/openpilot"
    if [ ! -d $openpilot_dir ]; then
        openpilot_dir=`find / -type d -name "openpilot" 2>/dev/null | head -n 1`
    fi
fi

package_dir=$openpilot_dir"/crosscompiled_package/"
dest_dir=$package_dir"openpilot/"

if [ -d $dest_dir ] 
then
    echo "Cleaning up previous package..." 
    rm -rf $dest_dir
fi

mkdir --parent "$dest_dir"

if [ -d $dest_dir ] 
then
    cd $openpilot_dir

    cp -v --parent `find -type f -executable -exec sh -c "file '{}' | grep -q 'aarch64'" \; -print` $dest_dir
    #cp -v --parent `find . -name "*.so"` $dest_dir
    cp --parent `find . -name "*.py"` $dest_dir
    cp --parent `find . -name "*.sh"` $dest_dir
    cp --parent `find . -name "*.capnp"` $dest_dir
    cp --parent `find . -name "service_list.yaml"` $dest_dir
    cp --parent `find . -name "*ui"` $dest_dir || true
    cp --parent `find . -name "version.h"` $dest_dir
    cp --parent `find . -name "setup_target_os.sh"` $dest_dir
    cp --parent `find . -name "loggerd"` $dest_dir || true
    cp cross-compilation-aarch64/Pipfile* $dest_dir
fi

pushd $dest_dir
ln -s rednose_repo/rednose rednose
popd

pushd $package_dir
ls -l -S -R openpilot/ > BOM.txt
tar -cvf openpilot.tar openpilot/
popd

if [ -d $dest_dir ] 
then
    echo "Cleaning up temp files..." 
    rm -rf $dest_dir
fi

echo ""
echo -e "\e[1;32m Installation package created successfully!! \e[0m"
echo -e "\e[1;33m --> $package_dir \e[0m"
echo ""