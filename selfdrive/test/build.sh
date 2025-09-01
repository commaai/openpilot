#!/usr/bin/env bash
#set -e
trap restore_root ERR
ORG_PWD="$PWD"

REPO="$HOME/work/openpilot/openpilot"
CACHE_ROOTFS_TARBALL_PATH="/tmp/rootfs_cache.tar"

unpack_rootfs_tarball() {
    cd /
    sudo tar -xf "$CACHE_ROOTFS_TARBALL_PATH" 2>/dev/null || true
    cd
}

commit_root() {
    sudo mkdir -p /base /newroot /upper /work
    
    sudo unshare -f --kill-child -m $ORG_PWD/selfdrive/test/build.sh build_inside_namespace
    ec=$?
    echo "end of ns"

    sudo rm -rf /base /newroot /work

    mkdir -p /tmp/rootfs_cache
    sudo rm -f "$CACHE_ROOTFS_TARBALL_PATH" # remove the old tarball from previous run, if exists
    cd /upper
    sudo tar -cf "$CACHE_ROOTFS_TARBALL_PATH" .
    cd

    sudo rm -rf /upper

    unpack_rootfs_tarball

    prepare_mounts

    exit $ec
}

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

restore_root() {
    echo failed at ${BASH_LINENO[0]}
}

build_inside_namespace() {
    mount --bind / /base
    mount -t overlay overlay -o lowerdir=/base,upperdir=/upper,workdir=/work /newroot
    rm -f /newroot/etc/resolv.conf
    touch /newroot/etc/resolv.conf
    cat /etc/resolv.conf > /newroot/etc/resolv.conf

    mkdir -p /newroot/old
    cd /newroot
    pivot_root . old

    mount -t proc proc /proc
    mount -t devtmpfs devtmpfs /dev
    mkdir -p /dev/pts
    mount -t devpts devpts /dev/pts
    mount -t proc proc /proc
    mount -t sysfs sysfs /sys

    touch /root_committed
    sudo -u runner /home/runner/work/openpilot/openpilot/selfdrive/test/build.sh
    ec=$?
    exit $ec
}

if [ "$1" = "build_inside_namespace" ]
then
    build_inside_namespace
    exit
fi

if [ -f "$CACHE_ROOTFS_TARBALL_PATH" ]
then
    # if the rootfs diff tarball (also created by this script) got restored from the CI native cache, unpack it, upgrading the rootfs
    echo "restoring rootfs from the native build cache"
    unpack_rootfs_tarball
    rm "$CACHE_ROOTFS_TARBALL_PATH"

    # before the next tasks are run, finalize the environment for them
    prepare_mounts

    # EXITS HERE - if the rootfs could been prepared entirely from the cache, there's no need for any further action like re-building
    exit 0
else
    # otherwise, we'll have to install everything from scratch and build the tarball to be available for the next run
    echo "no native build cache entry restored, rebuilding"
fi

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
sudo cp "/home/runner/work/openpilot/openpilot/pyproject.toml" "/home/runner/work/openpilot/openpilot/uv.lock" "/home/runner/work/openpilot/openpilot/tools/install_python_dependencies.sh" \
    /home/runner/

cd
rm -rf .venv

mkdir aaa
cd aaa
../install_python_dependencies.sh
cd
rm pyproject.toml uv.lock install_python_dependencies.sh


# add a git safe directory for compiling openpilot
sudo git config --global --add safe.directory /tmp/openpilot

# finally, create the rootfs diff tarball (to be pushed into the CI native cache)


# before the next tasks are run, finalize the environment for them
#prepare_mounts
