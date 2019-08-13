#!/bin/bash
set -e

docker build -t tmppilot -f Dockerfile.openpilot .

docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && make -C cereal && cd /tmp/openpilot/selfdrive/test/ && ./test_fingerprints.py'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && pyflakes $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda" | grep -vi "^\./tools")'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && pylint $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda" | grep -vi "^\./tools"); exit $(($? & 3))'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && make -C cereal && python -m unittest discover common'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && make -C cereal && python -m unittest discover selfdrive/can'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && make -C cereal && python -m unittest discover selfdrive/boardd'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && make -C cereal && python -m unittest discover selfdrive/controls'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && python -m unittest discover selfdrive/loggerd'
docker run --rm -v "$(pwd)"/selfdrive/test/tests/plant/out:/tmp/openpilot/selfdrive/test/tests/plant/out tmppilot /bin/sh -c 'cd /tmp/openpilot/selfdrive/test/tests/plant && OPTEST=1 ./test_longitudinal.py'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && make -C cereal && cd /tmp/openpilot/selfdrive/test/tests/process_replay/ && ./test_processes.py'
docker run --rm tmppilot /bin/sh -c 'mkdir -p /data/params && cd /tmp/openpilot/ && make -C cereal && cd /tmp/openpilot/selfdrive/test/ && ./test_car_models_openpilot.py'
