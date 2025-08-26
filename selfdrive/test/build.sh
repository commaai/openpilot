#!/usr/bin/env bash
#set -e
trap restore_roota ERR
ORG_PWD="$PWD"

REPO="$HOME/work/openpilot/openpilot"
CACHE_ROOTFS_TARBALL_PATH="/tmp/rootfs_cache.tar"

prepare_mounts() {
    # create and mount the required volumes where they're expected
    mkdir -p /tmp/openpilot /tmp/scons_cache /tmp/comma_download_cache /tmp/openpilot_cache
    sudo mount --bind "$REPO" /tmp/openpilot

    sudo mount --bind "$REPO/.ci_cache/scons_cache" /tmp/scons_cache || true
    sudo mount --bind "$REPO/.ci_cache/comma_download_cache" /tmp/comma_download_cache || true
    sudo mount --bind "$REPO/.ci_cache/openpilot_cache" /tmp/openpilot_cache || true

    # needed for the unit tests not to fail
    sudo chmod 755 /sys/fs/pstore
}

commit_root() {

    sudo mkdir -p /base /newroot /upper /work
    
    sudo unshare -f --kill-child -m $ORG_PWD/selfdrive/test/build_inside.sh
    ec=$?
    echo "end of ns"

    sudo rm -rf /base /newroot /work

    mkdir -p /tmp/rootfs_cache
    sudo rm -f "$CACHE_ROOTFS_TARBALL_PATH" # remove the old tarball from previous run, if exists
    cd /upper
    sudo tar -cf "$CACHE_ROOTFS_TARBALL_PATH" .
    cd

    sudo rm -rf /upper

    cd /
    sudo tar -xf "$CACHE_ROOTFS_TARBALL_PATH" 2>/dev/null || true
    cd

    prepare_mounts

    exit $ec


}

#declare -a mounts
commit_root_old() {

    sudo mkdir -p /upper /work /overlay # prepare directories for overlayfs
    org_mounts="$(cat /proc/mounts)" # save the original mounts table

    target="/overlay"
    sudo mkdir -p /lower
    sudo mount --bind / /lower
    mounts+=("/lower")
    sudo mount -t overlay overlay -o lowerdir=/lower,upperdir=/upper,workdir=/work "$target" # mount the overlayfs
    mounts+=("/")

    while read line # bind-mount any mounts under the old rootfs into the new one (overlayfs isn't recursive like e.g. `mount --rbind`)
    do
        echo DOING $line
        if [ "$line" != "/" ] # except the rootfs base, to avoid infinite mountpoint loops
        then
            target="/overlay$line"
            sudo mount --bind "$line" "$target" || true
            mounts+=("$line")
        fi
    done < <(echo "$org_mounts" | cut -d" " -f2)

    # remove the MS_SHARED flag from the original rootfs mount, which isn't supported by pivot_root(8) and would cause it to fail (see: https://lxc-users.linuxcontainers.narkive.com/pNQKxcnN/pivot-root-failures-when-is-mounted-as-shared)
    sudo mount --make-rprivate /

    # prepare for the pivot_root(8) and execute, swapping places of the original rootfs and the new one on overlayfs (with its lowerdir still being the original one)
    # (what this achieves is committing the state of the original rootfs and making it read-only, while creating a new, virtual read-write rootfs with all changes written into a separate directory, the upperdir)
    cd /overlay
    sudo mkdir -p old
    sudo pivot_root . old # once this finishes, the system is moved to the new rootfs and all newly open file descriptors will point to it
    #sudo systemctl daemon-reexec
    cd

    mount

    ls /home/runner

    sudo touch /root_commited

    exec "$ORG_PWD/$0" "$@"


}

restore_roota() {

    echo failed at ${BASH_LINENO[0]}

}

restore_root() {

    echo failed at ${BASH_LINENO[0]}

    echo mounts "${mounts[@]}"

    cd /old
    sudo mkdir -p new
    sudo pivot_root . new
    #sudo systemctl daemon-reexec
    cd


    for (( i=${#mounts[@]}-1; i>=0; i-- ))
    do
        #sudo lsof "/new/${mounts[i]}" || true
        sudo umount -l "/lower/overlay/${mounts[i]}" || true
        sudo umount -l "/new/${mounts[i]}" || true
    done

    mount

    rm -f /root_commited

}

echo AAA
mount
echo BBB

if [ -f "$CACHE_ROOTFS_TARBALL_PATH" ]
then
    # if the rootfs diff tarball (also created by this script) got restored from the CI native cache, unpack it, upgrading the rootfs
    echo "restoring rootfs from the native build cache"
    cd /
    sudo tar -xf "$CACHE_ROOTFS_TARBALL_PATH" 2>/dev/null || true
    cd
    rm "$CACHE_ROOTFS_TARBALL_PATH"

    # before the next tasks are run, finalize the environment for them
    prepare_mounts

    # EXITS HERE - if the rootfs could been prepared entirely from the cache, there's no need for any further action like re-building
    exit 0
else
    # otherwise, we'll have to install everything from scratch and build the tarball to be available for the next run
    echo "no native build cache entry restored, rebuilding"
fi

# in case this script was run on the same instance before, umount any overlays which were mounted by the previous runs
#tac /proc/mounts | grep overlay | grep -v "overlay / " | cut -d" " -f2 | while read line; do sudo umount "$line" || true; done

# in order to be able to build a diff rootfs tarball, we need to commit its initial state by moving it on-the-fly to overlayfs;
# below, we prepare the system and the new rootfs itself


if ! [ -e /root_committed ]
then
commit_root
fi



# -------- at this point, the original rootfs was committed and all the changes to it done below will be saved to the newly created rootfs diff tarball --------

# install and set up the native dependencies needed
PYTHONUNBUFFERED=1
DEBIAN_FRONTEND=noninteractive

mkdir -p /tmp/tools
cp "$REPO/tools/install_ubuntu_dependencies.sh" /tmp/tools/
sudo /tmp/tools/install_ubuntu_dependencies.sh &>/dev/null

sudo apt-get install -y --no-install-recommends \
    sudo tzdata locales ssh pulseaudio xvfb x11-xserver-utils gnome-screenshot python3-tk python3-dev \
    apt-utils alien unzip tar curl xz-utils dbus gcc-arm-none-eabi tmux vim libx11-6 wget &>/dev/null

sudo rm -rf /var/lib/apt/lists/*
sudo apt-get clean

cd /usr/lib/gcc/arm-none-eabi/*
sudo rm -rf arm/ thumb/nofp thumb/v6* thumb/v8* thumb/v7+fp thumb/v7-r+fp.sp
cd

sudo sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
sudo locale-gen
LANG=en_US.UTF-8
LANGUAGE=en_US:en
LC_ALL=en_US.UTF-8

mkdir -p /tmp/opencl-driver-intel
cd /tmp/opencl-driver-intel
wget https://github.com/intel/llvm/releases/download/2024-WW14/oclcpuexp-2024.17.3.0.09_rel.tar.gz &>/dev/null
wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.12.0/oneapi-tbb-2021.12.0-lin.tgz &>/dev/null
sudo mkdir -p /opt/intel/oclcpuexp_2024.17.3.0.09_rel
cd /opt/intel/oclcpuexp_2024.17.3.0.09_rel
sudo tar -zxvf /tmp/opencl-driver-intel/oclcpuexp-2024.17.3.0.09_rel.tar.gz
sudo mkdir -p /etc/OpenCL/vendors
sudo bash -c "echo /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd"
cd /opt/intel
sudo tar -zxvf /tmp/opencl-driver-intel/oneapi-tbb-2021.12.0-lin.tgz
sudo ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbb.so /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
sudo ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbbmalloc.so /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
sudo ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbb.so.12 /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
sudo ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbbmalloc.so.2 /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
sudo mkdir -p /etc/ld.so.conf.d
sudo bash -c "echo /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64 > /etc/ld.so.conf.d/libintelopenclexp.conf"
sudo ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
cd /
rm -rf /tmp/opencl-driver-intel
cd

sudo bash -c "dbus-uuidgen > /etc/machine-id"

NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
QTWEBENGINE_DISABLE_SANDBOX=1

# install and set up the Python dependencies needed

#sudo useradd -m -s /bin/bash -u 1002 test
sudo cp "/home/runner/work/openpilot/openpilot/pyproject.toml" "/home/runner/work/openpilot/openpilot/uv.lock" "/home/runner/work/openpilot/openpilot/tools/install_python_dependencies.sh" \
    /home/runner/
#sudo chown -R test:test /home/test

cd
#chown -R runner:run

rm -rf .venv

mkdir aaa
cd aaa
../install_python_dependencies.sh
cd


#sudo -u test bash -c " export HOME=/home/test ; export XDG_CONFIG_HOME=/home/test/.config ; cd /home/test ; ./install_python_dependencies.sh"
#sudo chown -R runner:runner /home/test

#cd /home/test ; rm pyproject.toml uv.lock install_python_dependencies.sh ; ls -la ; cd
rm pyproject.toml uv.lock install_python_dependencies.sh

#sudo rsync -aL /home/test/.venv/ /home/runner/.venv/
#sudo rsync -aL /home/test/.local/ /home/runner/.local/
#sudo rm -rf /home/test


# add a git safe directory for compiling openpilot
sudo git config --global --add safe.directory /tmp/openpilot

# finally, create the rootfs diff tarball (to be pushed into the CI native cache)


# before the next tasks are run, finalize the environment for them
#prepare_mounts
