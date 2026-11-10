# Builds the setup, reset and updater UIs into standalone executables with Nuitka.
# Ubuntu 20.04 sets the glibc floor (2.31) of the output.
FROM ubuntu:20.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++-10 ca-certificates curl capnproto libcapnp-dev patchelf \
      libdrm2 libssl-dev zlib1g-dev libffi-dev libbz2-dev liblzma-dev uuid-dev && \
    rm -rf /var/lib/apt/lists/*

# Nuitka links libpython statically, so the output doesn't depend on the OS's Python
ARG PYTHON_VERSION=3.12.12
RUN curl -fsSL https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz | tar xz -C /tmp && \
    cd /tmp/Python-$PYTHON_VERSION && \
    ./configure --prefix=/opt/python > /dev/null && \
    make -j$(nproc) > /dev/null && make install > /dev/null && \
    rm -rf /tmp/Python-$PYTHON_VERSION
ENV PATH=/opt/python/bin:$PATH

COPY tools/release/pack/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /src
WORKDIR /src
ARG ONEFILE=0
RUN ONEFILE=$ONEFILE tools/release/pack/compile.sh /out

# The apps must start without a Python install
FROM ubuntu:20.04 AS test
COPY --from=build /out /out
RUN cd /tmp && for app in setup reset updater; do \
      bin=/out/$app.dist/$app; [ -f /out/$app ] && bin=/out/$app; \
      env -i RAYLIB_BACKEND=headless LC_ALL=C HOME=/tmp timeout 10 "$bin" /bin/false /nonexistent > /tmp/$app.log 2>&1; \
      status=$?; if [ $status -ne 124 ]; then cat /tmp/$app.log; echo "$app exited with $status"; exit 1; fi; \
    done

FROM scratch
COPY --from=test /out /
