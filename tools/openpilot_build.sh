docker run --rm -v $(pwd):/tmp/openpilot -it commaai/openpilot bash -c 'cd /tmp/openpilot && scons -j$(nproc)'
