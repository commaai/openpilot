Tested in Ubuntu 16.04
(test machine has GPUs. may need some modification if have no GPU)

# How-To-Use:

# == 0. Have a working python 3.7.3 with pip ==

# == 1. Checkout openpilot ==
cd ~/
git clone -b sim https://github.com/commaai/openpilot.git

# == 2. Build openpilot ==
// [capnproto](https://capnproto.org/install.html) and capnpc-c may be needed for this step
cd ~/openpilot
scons

# == 3. Install environment from pipfile ==
(if have pipenv already, skip this step) pip install pipenv
cd ~/openpilot/tools/sim
pipenv install

# == 4. Download and unpack CARLA ==
// curl seems to randomly stop when getting big files, so we suggest downloading
// [CARLA](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.5.tar.gz)
// to ~/openpilot/tools/sim with a browser first then continue
cd ~/openpilot/tools/sim
./get_carla_095.sh

# == 5a. Start CARLA server (in terminal tap 1) ==
cd ~/openpilot/tools/sim
./run_carla_095.sh

# == 5b. Start openpilot (in terminal tap 2) ==
add 'export PYTHONPATH=$HOME/openpilot' to your ~/.bashrc
cd ~/openpilot/tools/sim
pipenv shell
cd ~/openpilot/selfdrive/
CUDA_VISIBLE_DEVICES='' PASSIVE=0 NOBOARD=1 ./manager.py

# == 5c. Start bridge (in terminal tap 3) ==
// links carla to openpilot, will "start the car" according to manager
cd ~/openpilot/tools/sim
pipenv shell
./bridge.py
// once started, you can do basic vehicle control, or engage openpilot with logitech G29
// you can also modify the code to drive with any other input method
