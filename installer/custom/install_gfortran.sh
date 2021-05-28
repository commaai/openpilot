#!/data/data/com.termux/files/usr/bin/sh
# Get some needed tools. coreutils for mkdir command, gnugp for the signing key, and apt-transport-https to actually connect to the repo
apt-get update
apt-get --assume-yes upgrade
apt-get --assume-yes install coreutils gnupg

# Make the sources.list.d directory
mkdir -p $PREFIX/etc/apt/sources.list.d

# Write the needed source file
echo "deb https://its-pointless.github.io/files/24 termux extras" > $PREFIX/etc/apt/sources.list.d/pointless.list

# Add signing key from https://its-pointless.github.io/pointless.gpg
curl -sL https://its-pointless.github.io/pointless.gpg | apt-key add -

# Update apt
apt update

# install gfortran
apt install gcc-11 -y
setupclang-gfort-11

# Elf cleaner is needed to remove a DT_ENTRY warning that prints out when gfortran -v is called to get 
# its version number and this breaks the pip installation script when fortran is used.

# Build elf cleaner
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ELFCLEANERPATH=$SCRIPTPATH/termux-elf-cleaner/
cd $ELFCLEANERPATH
make

# Perform elf cleaner on gfortran
./termux-elf-cleaner $(which gfortran)

