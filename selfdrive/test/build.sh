#!/usr/bin/env bash
set -e
ORG_PWD="$PWD"
SELF_PATH="$(realpath $0)"
SCRIPT_DIR="$(dirname "$SELF_PATH")"

REPO="$HOME/work/openpilot/openpilot"
CACHE_ROOTFS_TARBALL_PATH="/tmp/rootfs_cache.tar"

source "$SCRIPT_DIR/build_common.sh"

# if the rootfs diff tarball (also created by this script) got restored from the CI native cache
echo a
stat "$CACHE_ROOTFS_TARBALL_PATH" || true
ls "$CACHE_ROOTFS_TARBALL_PATH" || true
echo b
if [ -f "$CACHE_ROOTFS_TARBALL_PATH" ]
then
    # apply it, upgrading the rootfs
    echo "restoring rootfs from the native build cache"
    apply_rootfs_diff
    rm "$CACHE_ROOTFS_TARBALL_PATH"

    # before the next tasks are run, finalize the environment for them
    prepare_build

    # EXITS HERE - if the rootfs could been prepared entirely from the cache, there's no need for any further action like re-building
    exit 0
else
    # otherwise, we'll have to install everything from scratch and build the tarball to be available for the next run
    if ! [ -f /root_committed ]
    then
        echo "no native build cache entry restored, rebuilding"
    fi
fi

# in order to be able to build a diff rootfs tarball, we need to commit its initial state
# by moving it on-the-fly to overlayfs; below, we prepare the system and the new rootfs itself
commit_root

# -------- at this point, the original rootfs was committed and all the changes to it done below will be saved to the newly created rootfs diff tarball --------

# install and set up the native dependencies needed
PYTHONUNBUFFERED=1
DEBIAN_FRONTEND=noninteractive

mkdir -p /tmp/tools
cp "$REPO/tools/install_ubuntu_dependencies.sh" /tmp/tools/
sudo /tmp/tools/install_ubuntu_dependencies.sh

sudo apt-get install -y --no-install-recommends \
    sudo tzdata locales ssh pulseaudio xvfb x11-xserver-utils gnome-screenshot python3-tk python3-dev \
    apt-utils alien unzip tar curl xz-utils dbus gcc-arm-none-eabi tmux vim libx11-6 wget

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
rm -rf /tmp/opencl-driver-intel
cd

sudo bash -c "dbus-uuidgen > /etc/machine-id"

NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
QTWEBENGINE_DISABLE_SANDBOX=1

# install and set up the Python dependencies needed
cp "$REPO/pyproject.toml" "$REPO/uv.lock" "$HOME/"
mkdir "$HOME/tools"
cp "$REPO/tools/install_python_dependencies.sh" "$HOME/tools/"

VIRTUAL_ENV=/home/$USER/.venv
PATH="$VIRTUAL_ENV/bin:$PATH"

cd
tools/install_python_dependencies.sh
rm -rf tools/ pyproject.toml uv.lock .cache

# add a git safe directory for compiling openpilot
sudo git config --global --add safe.directory /tmp/openpilot
