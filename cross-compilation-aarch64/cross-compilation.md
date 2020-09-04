Cross-compiling Openpilot
----

A cross compiler is a compiler capable of creating executable code for a platform other than the one on which the compiler is running. In this case we will be generating ARM64 executables from a AMD64 machine

Pre-steps
----

Add ARM64 as a foreign architecture and update the source.list

---

```bash
sudo dpkg --add-architecture arm64
#now make sure the new platform was added...
sudo dpkg --print-foreign-architectures

sudo cp /etc/apt/sources.list /etc/apt/sources.list.bk
sudo gedit /etc/apt/sources.list
```

* Edit sources.list as follow:
add [arch=amd64] to all existing sources
```bash
deb [arch=amd64] http://ar.archive.ubuntu.com/ubuntu/ xenial-updates main restricted
# deb-src [arch=amd64] http://ar.archive.ubuntu.com/ubuntu/ xenial-updates main restricted
```

* Add the following repositories (within the same sources.list file as above):

```bash
deb [arch=arm64] http://ports.ubuntu.com/ xenial main restricted
deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates main restricted
deb [arch=arm64] http://ports.ubuntu.com/ xenial universe
deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates universe
deb [arch=arm64] http://ports.ubuntu.com/ xenial multiverse
deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates multiverse
deb [arch=arm64] http://ports.ubuntu.com/ xenial-backports main restricted universe multiverse
deb [arch=arm64] http://ppa.launchpad.net/keithw/glfw3/ubuntu trusty main
```

* Finally run:

```bash
sudo apt update
```

Install external dependencies and cross-compile libraries
----

Run `ubuntu_setup.sh -arch arm64` this will install all necessary dependencies.

* Reboot your system
* Clean your environment (scons --clean)
* After setting up ARM64, run ```scons cc=1``` in the Openpilot directory. 
```bash
cd /openpilot/
scons cc=1
```
NOTE: you can specify the number of CPU to speed up compilation i.e: ```scons cc=1 -j8```
