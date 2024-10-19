#!/usr/bin/env bash
set -eux

apt-get update
apt-get install -y debootstrap squashfs-tools

mkdir /tmp/chroot

# Install ubuntu into a chroot for us to later install our dependencies into.
debootstrap --variant=minbase noble /tmp/chroot http://archive.ubuntu.com/ubuntu
# The devices cause untar issues and we don't need them.
rm -rf /tmp/chroot/dev/*

cat <<EOF > /tmp/chroot/etc/apt/sources.list
deb http://archive.ubuntu.com/ubuntu noble main
deb http://archive.ubuntu.com/ubuntu noble universe
deb http://archive.ubuntu.com/ubuntu noble multiverse
deb http://archive.ubuntu.com/ubuntu noble restricted
EOF

cat <<EOF >> /tmp/chroot/etc/profile
DEBIAN_FRONTEND=noninteractive 
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
QTWEBENGINE_DISABLE_SANDBOX=1
LANG=en_US.UTF-8
LANGUAGE=en_US:en
LC_ALL=en_US.UTF-8
PYTHONUNBUFFERED=1
LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu
EOF

cat <<EOF > /tmp/chroot/tmp/install-deps.sh
#!/usr/bin/env bash
set -eux

apt-get update
apt-get install -y --no-install-recommends \
  sudo \
  tzdata \
  locales \
  ssh \
  pulseaudio \
  xvfb \
  x11-xserver-utils \
  gnome-screenshot \
  apt-utils \
  alien \
  unzip \
  tar \
  curl \
  xz-utils \
  dbus \
  gcc-arm-none-eabi \
  tmux \
  vim \
  libx11-6 \
  wget

dbus-uuidgen > /etc/machine-id

sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
locale-gen

cd /tmp/openpilot
./tools/ubuntu_setup.sh

mkdir -p /tmp/opencl-driver-intel
cd /tmp/opencl-driver-intel
wget https://github.com/intel/llvm/releases/download/2024-WW14/oclcpuexp-2024.17.3.0.09_rel.tar.gz
wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.12.0/oneapi-tbb-2021.12.0-lin.tgz
mkdir -p /opt/intel/oclcpuexp_2024.17.3.0.09_rel
cd /opt/intel/oclcpuexp_2024.17.3.0.09_rel
tar -zxvf /tmp/opencl-driver-intel/oclcpuexp-2024.17.3.0.09_rel.tar.gz
mkdir -p /etc/OpenCL/vendors
echo /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd
cd /opt/intel
tar -zxvf /tmp/opencl-driver-intel/oneapi-tbb-2021.12.0-lin.tgz
ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbb.so /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbbmalloc.so /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbb.so.12 /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbbmalloc.so.2 /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64
mkdir -p /etc/ld.so.conf.d 
echo /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64 > /etc/ld.so.conf.d/libintelopenclexp.conf
ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
cd /
rm -rf /tmp/opencl-driver-intel

# Remove arm architecture toolchains that we don't want.
cd /usr/lib/gcc/arm-none-eabi/*
rm -rf arm/ thumb/nofp thumb/v6* thumb/v8* thumb/v7+fp thumb/v7-r+fp.sp

git config --global --add safe.directory /tmp/openpilot

# Remove all tmp files except 'openpilot' directory and its contents
find /tmp -mindepth 1 -path '/tmp/openpilot' -prune -o -exec rm -rf {} +

# Remove cached apt archives.
apt clean
EOF
chmod +x /tmp/chroot/tmp/install-deps.sh

cat <<EOF > /tmp/chroot/run_ci.sh
#!/usr/bin/env bash
set -eux
env
cd /tmp/openpilot
ldd ./.venv/bin/python
. ./.venv/bin/activate
bash -c "\$1"
EOF
chmod +x /tmp/chroot/run_ci.sh

cp /etc/resolv.conf /tmp/chroot/resolv.conf
mount --bind /proc /tmp/chroot/proc
mount --bind /sys /tmp/chroot/sys
mount --bind /dev /tmp/chroot/dev
mkdir /tmp/chroot/tmp/openpilot
mount --bind "$GITHUB_WORKSPACE" /tmp/chroot/tmp/openpilot
chroot /tmp/chroot bash /tmp/install-deps.sh
umount /tmp/chroot/tmp/openpilot
umount /tmp/chroot/proc
umount /tmp/chroot/sys
umount /tmp/chroot/dev
cd /tmp/chroot
# A squashfs is faster than docker and we don't need to decompress unused files.
mksquashfs . /tmp/chroot.squashfs -b 256k -comp zstd -Xcompression-level 1 
cd /
rm -rf /tmp/chroot