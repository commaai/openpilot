# Openpilot build steps:
=========================
1. Install git:
    apt-get update
    apt-get install git

2. Clone openpilot:
    git clone --recurse-submodules https://github.com/commaai/openpilot.git

3. Install dependencies:
    apt-get install sudo
    apt-get install curl
    execute: ./openpilot/tools/ubuntu_setup.sh

4. Set up SCons
    pip3 install SCons      (must be version 4.0.1+)
    pip3 install numpy jinja2 Cython sympy cffi
    run scons command in openpilot

5. You're done

